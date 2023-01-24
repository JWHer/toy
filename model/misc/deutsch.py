import spacy
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self):
        self.lang_en = spacy.load('en_core_web_sm')
        self.lang_de = spacy.load('de_core_news_sm')
        self.tokenize_en = lambda text: [token.text for token in self.lang_en.tokenizer(text)]
        self.tokenize_de = lambda text: [token.text for token in self.lang_de.tokenizer(text)]
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        print('dataset initializing start')

    def make_dataset(self):
        self.source = Field(tokenize=self.tokenize_en, init_token=self.sos_token, eos_token=self.eos_token,
                            lower=True, batch_first=True)
        self.target = Field(tokenize=self.tokenize_de, init_token=self.sos_token, eos_token=self.eos_token,
                            lower=True, batch_first=True)

        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator

if __name__ == '__main__':
    tokenizer = Tokenizer()
    loader = DataLoader(ext=('.en', '.de'),
                        tokenize_en=tokenizer.tokenize_en,
                        tokenize_de=tokenizer.tokenize_de,
                        init_token='<sos>',
                        eos_token='<eos>')

    train, valid, test = loader.make_dataset()
    loader.build_vocab(train_data=train, min_freq=2)
    train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                        batch_size=batch_size,
                                                        device=device)

    src_pad_idx = loader.source.vocab.stoi['<pad>']
    trg_pad_idx = loader.target.vocab.stoi['<pad>']
    trg_sos_idx = loader.target.vocab.stoi['<sos>']

    enc_voc_size = len(loader.source.vocab)
    dec_voc_size = len(loader.target.vocab)
