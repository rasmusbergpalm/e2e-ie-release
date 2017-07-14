class IOBChunker(object):
    @staticmethod
    def chunk(words, labels):
        """
        Joins sequential and matching IOB labeled words into single labeled rects.

        :param words: List of words (strings).
        :param labels: Lit of labels (strings) in IOB notation, e.g. ['O','B-foo','I-foo']
        :return: list of tuple (class, n-gram). An n-gram is one or more words joined together
        """
        chunks = []
        current = []

        assert len(words) == len(labels)

        for word, label in zip(words, labels):
            if label == 'O':
                position = classification = 'O'
            else:
                position, classification = label.split("-")

            word = {'text': word, 'class': classification}

            if position == 'O':
                if current:
                    chunks.append(current)
                    current = []

            elif position == 'B':
                if current:
                    chunks.append(current)
                    current = []

                current.append(word)

            elif position == 'I':
                if current:
                    if classification == current[0]['class']:
                        current.append(word)
                    else:
                        chunks.append(current)
                        current = []

        if current:
            chunks.append(current)

        output = []
        for chunk in chunks:
            text = " ".join([w['text'] for w in chunk])
            _class = chunk[0]['class']

            output.append((_class, text))

        return output

    @staticmethod
    def to_object(words, labels, keys):
        """
        :param words: List of words (strings).
        :param labels: Lit of labels (strings) in IOB notation, e.g. ['O','B-foo','I-foo']
        :param keys: set of keys to include in object
        """

        def in_keys_predicate(x):
            return x[0] in keys

        a_chunks = list(filter(in_keys_predicate, IOBChunker.chunk(words, labels)))

        obj = {}
        for k, v in a_chunks:
            if k not in obj:
                obj[k] = v
            else:
                obj[k] += " , " + v

        return obj
