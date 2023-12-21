from transformers import BertForSequenceClassification, AutoTokenizer
import torch

class SentenceTypeClassifier:
    _bert:BertForSequenceClassification = None
    _tokenizer:AutoTokenizer = None
    _token_file:str = None
    _pretrained_classifier:str = None
    _device:torch.device = None

    def __init__(self, pretrained_classifier, token_file):
        self._pretrained_classifier = pretrained_classifier
        self._token_file = token_file

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device
    
    @property
    def token(self):
        return open(self._token_file, 'r').read()
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained_classifier, token = self.token)
        return self._tokenizer
    
    @property
    def bert(self):
        if self._bert is None:
            self._bert = BertForSequenceClassification.from_pretrained(self._pretrained_classifier, token=self.token).to(self.device)
            self._bert.eval()
        return self._bert

    def _get_classification_inputs(self, message):
        tokens = self.tokenizer(message, return_tensors='pt', max_length=400, padding='max_length', truncation=True)

        input_ids = tokens['input_ids'].to(self.device)
        attn_mask = tokens['attention_mask'].to(self.device)

        return input_ids, attn_mask
    
    def classify(self, message):
        input_ids, attn_mask = self._get_classification_inputs(message)
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        return preds

