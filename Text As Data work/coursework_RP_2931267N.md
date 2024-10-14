# 1. Background

One of the complexities of natural language is that in everyday conversations, people do not simply answer "yes" or "no" as machines might. The answers provided by people can be ambiguous or indirect. This requires systems to understand not just the literal meaning, but also the deeper implications—a core issue explored in this paper.

In past research, studies on indirect answers were based on small-scale data, with the majority of research focusing on direct answers. However, in the current mainstream,  indirect answers are becoming increasingly important and frequent. Yet, there is a lack of a large-scale corpus dedicated to such answers in the market.

# 2. Contributions

- **Creation of the "Circa" Dataset:** The authors invited native English speakers for crowdsourcing tasks, ensuring dataset quality and diversity with well-designed assignments and detailed annotations. This resulted in a dataset with 34,268 pairs of questions and indirect answers, covering various response types, including explicit yes/no, uncertain, intermediate, and conditional responses.
- **BERT-based Model:** Using a pre-trained BERT model, the authors explored transfer learning's potential in NLP tasks, particularly for understanding indirect answers in dialogue systems. They employed different experimental setups (fine-tuning, using varied models, and comparison references) and transfer learning strategies, enhancing model performance in classifying indirect answers.
- **Extensive Experimental Evaluation:** The authors conducted an experimental evaluation to assess fine-tuning BERT models for classifying indirect answers. Key findings highlight the need to consider question and answer texts for accurate classification and the performance gains from fine-tuning and transfer learning.

# 3. Critique


I believe the results somewhat support their conclusions. Through experimental evaluation, they demonstrated the effectiveness of fine-tuning pre-trained models in understanding indirect answers, particularly highlighting the potential of transfer learning strategies in enhancing model performance. However, the authors also acknowledge some deficiencies in their research, such as the loss of implicit information when handling additional information (like expressions of preference), indicating significant room for growth in data modeling.

My assessment of their research method is twofold:

- Positives: The new indirect answer dataset is indeed a significant contribution. It is based on real social dialogue scenarios, and the authors' data preprocessing methods are appropriately comprehensive, covering most daily conversational content, providing new perspectives and data support for studying indirect answers.
- Negatives: Negatives: The authors acknowledge the model's limited generalizability beyond the training scenarios and its struggle with additional information in indirect answers. Language evolution, including internet slang, poses further challenges for model adaptability, suggesting areas for future research. Additionally, the crowdsourcing participants were predominantly from the United States, overlooking regional English variations, such as UK dialects or slang, which could significantly affect indirect response interpretation. 