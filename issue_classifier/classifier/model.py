from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AlbertForSequenceClassification, AlbertTokenizer
import torch
import torch.nn.functional as F
import numpy as np


class AlbertForMultilabelSequenceClassification(AlbertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
                                        logits=logits,
                                        hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


class Model:
    def __init__(self):

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.labels = ['Bug', 'Improvement', 'New Feature']
        self.tokenizer = AlbertTokenizer.from_pretrained(
            'albert-base-v2', do_lower_case=True)
        classifier = AlbertForMultilabelSequenceClassification.from_pretrained(
            'albert-base-v2',
            output_attentions=False,
            output_hidden_states=False,
            num_labels=3
        )
        classifier.load_state_dict(
            torch.load("assets/pytorch_model.bin", map_location=self.device))
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='longest',
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,

        )

        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = self.classifier(input_ids, attention_mask)

        prediction = F.softmax(probabilities.logits,
                               dim=1).cpu().numpy().flatten().max()
        prediction_index = np.where(F.softmax(probabilities.logits,
                                              dim=1).cpu().numpy() == prediction)[1][0]
        label = self.labels[prediction_index]

        all_predictions = F.softmax(
            probabilities.logits, dim=1).cpu().numpy().flatten()

        bug_prediction = all_predictions[0]
        improvement_prediction = all_predictions[1]
        newFeature_prediction = all_predictions[2]
        # print(F.softmax(all_predictions)
        # print(prediction)
        # print(self.labels[prediction_index])
        # print(prediction_index)

        return (bug_prediction, improvement_prediction, newFeature_prediction)


model = Model()
# model.predict("this is an impsorvement")


def get_model():
    return model
