# NLP
This is a dedicated repository focusing on NLP , it's techniques and examples. Ihere is a complete roadmap for knowing NLP.

# TEXT PROCESSING LEVEL-1
->>   **Tokeization :**  Tokenization is a fundamental step in natural language processing (NLP) that involves breaking down a text into smaller units called tokens. These tokens can be words, subwords, or characters. Tokenization is essential because it prepares the text for further processing and analysis.

-->   **Types of Tokenization**

#Word Tokenization: Splitting text into individual words.

#Subword Tokenization: Splitting text into smaller units like subwords or morphemes.

#Character Tokenization: Splitting text into individual characters.

-->   **Tokenization Techniques**

#Whitespace Tokenization: Splits the text based on whitespace (spaces, tabs, newlines).

#Punctuation-based Tokenization: Splits the text based on punctuation marks.

#Regex Tokenization: Uses regular expressions to define patterns for splitting the text.

#Library-based Tokenization: Utilizes specialized libraries that implement advanced tokenization algorithms.

--> **Lemmatization Techniques**

Lemmatization techniques in natural language processing (NLP) involve methods to identify and transform words into their base or root forms, known as lemmas. These approaches contribute to text normalization, facilitating more accurate language analysis and processing in various NLP applications. Three types of lemmatization techniques are:

1. Rule Based Lemmatization:Rule-based lemmatization involves the application of predefined rules to derive the base or root form of a word. Unlike machine learning-based approaches, which learn from data, rule-based lemmatization relies on linguistic rules and patterns.
Here’s a simplified example of rule-based lemmatization for English verbs:
Rule: For regular verbs ending in “-ed,” remove the “-ed” suffix.
Example:
                  Word: “walked”
                  Rule Application: Remove “-ed”
                  Result: “walk
This approach extends to other verb conjugations, providing a systematic way to obtain lemmas for regular verbs. While rule-based lemmatization may not cover all linguistic nuances, it serves as a transparent and interpretable method for deriving base forms in many cases.

2. Dictionary-Based Lemmatization:Dictionary-based lemmatization relies on predefined dictionaries or lookup tables to map words to their corresponding base forms or lemmas. Each word is matched against the dictionary entries to find its lemma. This method is effective for languages with well-defined rules.
Suppose we have a dictionary with lemmatized forms for some words:

                                                                          ‘running’ -> ‘run’
                                                                          ‘better’ -> ‘good’
                                                                          ‘went’ -> ‘go’
When we apply dictionary-based lemmatization to a text like “I was running to become a better athlete, and then I went home,” the resulting lemmatized form would be: “I was run to become a good athlete, and then I go home.”

3. Machine Learning-Based Lemmatization: Machine learning-based lemmatization leverages computational models to automatically learn the relationships between words and their base forms. Unlike rule-based or dictionary-based approaches, machine learning models, such as neural networks or statistical models, are trained on large text datasets to generalize patterns in language.
Example: Consider a machine learning-based lemmatizer trained on diverse texts. When encountering the word ‘went,’ the model, having learned patterns, predicts the base form as ‘go.’ Similarly, for ‘happier,’ the model deduces ‘happy’ as the lemma. The advantage lies in the model’s ability to adapt to varied linguistic nuances and handle irregularities, making it robust for lemmatizing diverse vocabularies.




