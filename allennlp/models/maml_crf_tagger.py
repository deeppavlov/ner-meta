from typing import Dict, Optional
from collections import defaultdict, Iterable
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import SpanBasedF1Measure
import numpy as np

@Model.register("maml_crf_tagger")
class MAMLCrfTagger(Model):
    """
    The ``CrfTagger`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    constraint_type : ``str``, optional (default=``None``)
        If provided, the CRF will be constrained at decoding time
        to produce valid labels based on the specified type (e.g. "BIO", or "BIOUL").
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: list,
                 encoder: list,
                 label_namespace: str = "labels",
                 constraint_type: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.losses = []
        self.copy_parameters = [[] for _ in range(len(encoder))]
        self.n_copies = len(self.copy_parameters) - 1

        self.label_namespace = label_namespace

        self.text_field_embedders = text_field_embedder
#        self.text_field_embedder = text_field_embedder[-1]
#        for i in range(len(text_field_embedder)):
#           self.copy_parameters[i] += [w for w in text_field_embedder[i].parameters()]

        self.num_tags = self.vocab.get_vocab_size(label_namespace)

        self.encoders = encoder
        self.encoder = encoder[-1]
        for i in range(len(encoder)):
            self.copy_parameters[i] += [w for w in encoder[i].parameters()]

        for em in self.text_field_embedders:
            em.cuda(1)
        
        self.tag_projection_layers = [TimeDistributed(Linear(self.encoders[0].get_output_dim(),
                                                           self.num_tags)) for _ in range(len(encoder))]
        self.tag_projection_layer = self.tag_projection_layers[-1]
        for i in range(len(self.tag_projection_layers)):
            self.copy_parameters[i] += [w for w in self.tag_projection_layers[i].parameters()]

        if constraint_type is not None:
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(constraint_type, labels)
        else:
            constraints = None

        self.crfs = [ConditionalRandomField(self.num_tags, constraints) for _ in range(len(encoder))]
        self.crf = self.crfs[-1]
        for i in range(len(self.crfs)):
            self.copy_parameters[i] += [w for w in self.crfs[i].parameters()]
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace=label_namespace)

        check_dimensions_match(text_field_embedder[0].get_output_dim(), encoder[0].get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        initializer(self)
        self.optimizers = [torch.optim.Adam(self.copy_parameters[i], 0.0001) for i in range(self.n_copies+1)]


    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:

        logits : ``torch.FloatTensor``
            The logits that are the output of the ``tag_projection_layer``
        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[str]]``
            The predicted tags using the Viterbi algorithm.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        """
#        print([w.shape for w in self.parameters()])
#        print(self.state_dict())
#        1/0
        # print([int(w.sum().data.numpy()) for w in self.parameters()])
        # for i in range(self.n_copies+1):
        #     print([int(w.sum().data.numpy()) for w in self.copy_parameters[i]])
        # print('\n')
        if tokens['tokens'].volatile:
            embedded_text_input = self.text_field_embedders[-1](tokens)
            mask = util.get_text_field_mask(tokens)
            encoded_text = self.encoder(embedded_text_input, mask)

            logits = self.tag_projection_layer(encoded_text)
            predicted_tags = self.crf.viterbi_tags(logits, mask)

            output = {"logits": logits, "mask": mask, "tags": predicted_tags}
            if tags is not None:
                # Add negative log-likelihood as loss
                log_likelihood = self.crf(logits, tags, mask)
                output["loss"] = -log_likelihood/tags.shape[0]

                # Represent viterbi tags as "class probabilities" that we can
                # feed into the `span_metric`
                class_probabilities = logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1

                self.span_metric(class_probabilities, tags, mask)

            return output

        for i in range(self.n_copies):
            for w, w_main in zip(self.copy_parameters[i], self.copy_parameters[self.n_copies]):
                w.data = w_main.detach().data.clone()
                w.detach()
        losses_new = []
        losses_old = []
        f1_old = []
        f1_new = []
        n_steps = 10
        for i in range(self.n_copies):
            for group in self.optimizers[i].param_groups:
                for p in group['params']:
                     self.optimizers[i].state[p] = defaultdict(dict)

            for step in range(n_steps):
                self.optimizers[i].zero_grad()
                tokens_i = dict(zip(tokens.keys(), [tensor[i][32*step:32*(step+1)] for tensor in tokens.values()]))
                tags_i = tags[i][32*step:32*(step+1)].clone()

                embedded_text_input = self.text_field_embedders[i](tokens_i)
                mask = util.get_text_field_mask(tokens_i)
                encoded_text = self.encoders[i](embedded_text_input, mask)

                logits = self.tag_projection_layers[i](encoded_text)
                predicted_tags = self.crfs[i].viterbi_tags(logits, mask)

                # Add negative log-likelihood as loss
                log_likelihood = self.crfs[i](logits, tags_i, mask)
                loss = -log_likelihood/tags_i.shape[0]
                # Represent viterbi tags as "class probabilities" that we can
                # feed into the `span_metric`
                class_probabilities = logits * 0.
                for index, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[index, j, tag_id] = 1
                if step == 0:
                    self.span_metric(class_probabilities, tags_i, mask)
                    losses_old.append(loss.data.cpu().numpy())
                    f1_old.append(self.get_metrics()['f1-measure-overall'])

                loss.backward()
                self.optimizers[i].step()

            # Last
            self.optimizers[i].zero_grad()

            tokens_i = dict(zip(tokens.keys(), [tensor[i][-32:] for tensor in tokens.values()]))
            tags_i = tags[i][-32:].clone()

            embedded_text_input = self.text_field_embedders[i](tokens_i)
            mask = util.get_text_field_mask(tokens_i)
            encoded_text = self.encoders[i](embedded_text_input, mask)

            logits = self.tag_projection_layers[i](encoded_text)
            predicted_tags = self.crfs[i].viterbi_tags(logits, mask)

            if tags is not None:
                # Add negative log-likelihood as loss
                log_likelihood = self.crfs[i](logits, tags_i, mask)
                loss = -log_likelihood/tags_i.shape[0]
                # Represent viterbi tags as "class probabilities" that we can
                # feed into the `span_metric`
                class_probabilities = logits * 0.
                for index, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[index, j, tag_id] = 1
                # if i==0:
                #     print(logits[0])
                self.span_metric(class_probabilities, tags_i, mask)
            losses_new.append(loss.data.cpu().numpy())

            f1_new.append(self.get_metrics()['f1-measure-overall'])

            loss.backward()

        for i_copy in range(self.n_copies):
            for i_param, w in enumerate(self.parameters()):
                if i_copy > 0:
                    w.grad += self.copy_parameters[i_copy][i_param].grad.clone().detach()/self.n_copies
                else:
                    w.grad = self.copy_parameters[i_copy][i_param].grad.clone().detach()/self.n_copies
                    #w.grad = self.copy_parameters[i_copy][i_param].grad.clone().detach() * 0

        #output = {"logits": logits, "mask": mask, "tags": predicted_tags}
        output = {}
        if tags is not None:
            # Add negative log-likelihood as loss
            #log_likelihood = self.crf(logits, tags, mask)
            device = self.copy_parameters[0][0].get_device()
            output["loss"] = torch.autograd.Variable(torch.FloatTensor(np.array([np.mean(losses_new)]))).cuda(device)
            self.losses.append(np.mean(losses_new))
            np.save('losses', self.losses)
            print('\n', np.mean(losses_old), np.mean(losses_new), '\n')
            print('\n', np.mean(f1_old), np.mean(f1_new), '\n')

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the `span_metric`
            class_probabilities = logits * 0.
            # for i, instance_tags in enumerate(predicted_tags):
            #     for j, tag_id in enumerate(instance_tags):
            #         class_probabilities[i, j, tag_id] = 1
            #
            # self.span_metric(class_probabilities, tags, mask)

        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace="labels")
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = self.span_metric.get_metric(reset=True)
        # self.precisions.append(metric_dict['f1-measure-overall'])
        # np.save('f1_basic', self.precisions)
        return {x: y for x, y in metric_dict.items() if "overall" in x}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'MAMLCrfTagger':
        embedder_params = params.pop("text_field_embedder")
        encoder_params = params.pop("encoder")
        text_field_embedder = []
        encoder = []
        label_namespace = params.pop("label_namespace", "labels")
        constraint_type = params.pop("constraint_type", None)
        initializer_params = params.pop('initializer', [])
        reg_params = params.pop('regularizer', [])
        for i in range(20 + 1):
            print(i)
            encoder.append(Seq2SeqEncoder.from_params(encoder_params.duplicate()))
#            device = [w for w in encoder[-1].parameters()][0].get_device()
            text_field_embedder.append(TextFieldEmbedder.from_params(vocab, embedder_params.duplicate()))
        initializer = InitializerApplicator.from_params(initializer_params)
        regularizer = RegularizerApplicator.from_params(reg_params)
        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   label_namespace=label_namespace,
                   constraint_type=constraint_type,
                   initializer=initializer,
                   regularizer=regularizer)
