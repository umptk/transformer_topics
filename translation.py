from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


# IMPORT MODEL FOR TRANSLATION AND TOKENIZATION, UNPICKLE DOCS
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


# FUNCTION FOR TRANSLATING A PICKLED docs LIST TO TARGET LANG.
def many_to_many(docs: list, target_lang: str, source_lang: str='en_XX'):

    target_docs = list()

    for doc in docs:
        tokenizer.src_lang = source_lang
        encoded_hi = tokenizer(doc, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded_hi,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
        )
        target_docs.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])

    return target_docs


# target_docs = many_to_many(docs_samp[1:2], 'ko_KR')
