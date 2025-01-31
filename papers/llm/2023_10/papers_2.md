# llm - 2023_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.03659v1">Balancing Autonomy and Alignment: A Multi-Dimensional Taxonomy for Autonomous LLM-powered Multi-Agent Architectures</a></div>
    <div class="paper-meta">
      📅 2023-10-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized the field of artificial intelligence, endowing it with sophisticated language understanding and generation capabilities. However, when faced with more complex and interconnected tasks that demand a profound and iterative thought process, LLMs reveal their inherent limitations. Autonomous LLM-powered multi-agent systems represent a strategic response to these challenges. Such systems strive for autonomously tackling user-prompted goals by decomposing them into manageable tasks and orchestrating their execution and result synthesis through a collective of specialized intelligent agents. Equipped with LLM-powered reasoning capabilities, these agents harness the cognitive synergy of collaborating with their peers, enhanced by leveraging contextual resources such as tools and datasets. While these architectures hold promising potential in amplifying AI capabilities, striking the right balance between different levels of autonomy and alignment remains the crucial challenge for their effective operation. This paper proposes a comprehensive multi-dimensional taxonomy, engineered to analyze how autonomous LLM-powered multi-agent systems balance the dynamic interplay between autonomy and alignment across various aspects inherent to architectural viewpoints such as goal-driven task management, agent composition, multi-agent collaboration, and context interaction. It also includes a domain-ontology model specifying fundamental architectural concepts. Our taxonomy aims to empower researchers, engineers, and AI practitioners to systematically analyze the architectural dynamics and balancing strategies employed by these increasingly prevalent AI systems. The exploratory taxonomic classification of selected representative LLM-powered multi-agent systems illustrates its practical utility and reveals potential for future research and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.13230v2">To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis</a></div>
    <div class="paper-meta">
      📅 2023-10-05
      | 💬 Accepted at NeurIPS 2023
    </div>
    <details class="paper-abstract">
      Recent research has highlighted the importance of dataset size in scaling language models. However, large language models (LLMs) are notoriously token-hungry during pre-training, and high-quality text data on the web is approaching its scaling limit for LLMs. To further enhance LLMs, a straightforward approach is to repeat the pre-training data for additional epochs. In this study, we empirically investigate three key aspects under this approach. First, we explore the consequences of repeating pre-training data, revealing that the model is susceptible to overfitting, leading to multi-epoch degradation. Second, we examine the key factors contributing to multi-epoch degradation, finding that significant factors include dataset size, model parameters, and training objectives, while less influential factors consist of dataset quality and model FLOPs. Finally, we explore whether widely used regularization can alleviate multi-epoch degradation. Most regularization techniques do not yield significant improvements, except for dropout, which demonstrates remarkable effectiveness but requires careful tuning when scaling up the model size. Additionally, we discover that leveraging mixture-of-experts (MoE) enables cost-effective and efficient hyper-parameter tuning for computationally intensive dense LLMs with comparable trainable parameters, potentially impacting efficient LLM development on a broader scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.00436v3">SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning</a></div>
    <div class="paper-meta">
      📅 2023-10-05
    </div>
    <details class="paper-abstract">
      The recent progress in large language models (LLMs), especially the invention of chain-of-thought prompting, has made it possible to automatically answer questions by stepwise reasoning. However, when faced with more complicated problems that require non-linear thinking, even the strongest LLMs make mistakes. To address this, we explore whether LLMs are able to recognize errors in their own step-by-step reasoning, without resorting to external resources. To this end, we propose SelfCheck, a general-purpose zero-shot verification schema for recognizing such errors. We then use the results of these checks to improve question-answering performance by conducting weighted voting on multiple solutions to the question. We test SelfCheck on three datasets (GSM8K, MathQA, and MATH) and find that it successfully recognizes errors and, in turn, increases final answer accuracies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.03414v1">LLM Based Multi-Document Summarization Exploiting Main-Event Biased Monotone Submodular Content Extraction</a></div>
    <div class="paper-meta">
      📅 2023-10-05
    </div>
    <details class="paper-abstract">
      Multi-document summarization is a challenging task due to its inherent subjective bias, highlighted by the low inter-annotator ROUGE-1 score of 0.4 among DUC-2004 reference summaries. In this work, we aim to enhance the objectivity of news summarization by focusing on the main event of a group of related news documents and presenting it coherently with sufficient context. Our primary objective is to succinctly report the main event, ensuring that the summary remains objective and informative. To achieve this, we employ an extract-rewrite approach that incorporates a main-event biased monotone-submodular function for content selection. This enables us to extract the most crucial information related to the main event from the document cluster. To ensure coherence, we utilize a fine-tuned Language Model (LLM) for rewriting the extracted content into a coherent text. The evaluation using objective metrics and human evaluators confirms the effectiveness of our approach, as it surpasses potential baselines, demonstrating excellence in both content coverage, coherence, and informativeness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.03016v1">Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions</a></div>
    <div class="paper-meta">
      📅 2023-10-04
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      In order to understand the in-context learning phenomenon, recent works have adopted a stylized experimental framework and demonstrated that Transformers can learn gradient-based learning algorithms for various classes of real-valued functions. However, the limitations of Transformers in implementing learning algorithms, and their ability to learn other forms of algorithms are not well understood. Additionally, the degree to which these capabilities are confined to attention-based models is unclear. Furthermore, it remains to be seen whether the insights derived from these stylized settings can be extrapolated to pretrained Large Language Models (LLMs). In this work, we take a step towards answering these questions by demonstrating the following: (a) On a test-bed with a variety of Boolean function classes, we find that Transformers can nearly match the optimal learning algorithm for 'simpler' tasks, while their performance deteriorates on more 'complex' tasks. Additionally, we find that certain attention-free models perform (almost) identically to Transformers on a range of tasks. (b) When provided a teaching sequence, i.e. a set of examples that uniquely identifies a function in a class, we show that Transformers learn more sample-efficiently. Interestingly, our results show that Transformers can learn to implement two distinct algorithms to solve a single task, and can adaptively select the more sample-efficient algorithm depending on the sequence of in-context examples. (c) Lastly, we show that extant LLMs, e.g. LLaMA-2, GPT-4, can compete with nearest-neighbor baselines on prediction tasks that are guaranteed to not be in their training set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.02238v2">Who's Harry Potter? Approximate Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on massive internet corpora that often contain copyrighted content. This poses legal and ethical challenges for the developers and users of these models, as well as the original authors and publishers. In this paper, we propose a novel technique for unlearning a subset of the training data from a LLM, without having to retrain it from scratch. We evaluate our technique on the task of unlearning the Harry Potter books from the Llama2-7b model (a generative language model recently open-sourced by Meta). While the model took over 184K GPU-hours to pretrain, we show that in about 1 GPU hour of finetuning, we effectively erase the model's ability to generate or recall Harry Potter-related content, while its performance on common benchmarks (such as Winogrande, Hellaswag, arc, boolq and piqa) remains almost unaffected. We make our fine-tuned model publicly available on HuggingFace for community evaluation. To the best of our knowledge, this is the first paper to present an effective technique for unlearning in generative language models. Our technique consists of three main components: First, we use a reinforced model that is further trained on the target data to identify the tokens that are most related to the unlearning target, by comparing its logits with those of a baseline model. Second, we replace idiosyncratic expressions in the target data with generic counterparts, and leverage the model's own predictions to generate alternative labels for every token. These labels aim to approximate the next-token predictions of a model that has not been trained on the target data. Third, we finetune the model on these alternative labels, which effectively erases the original text from the model's memory whenever it is prompted with its context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.03046v1">EcoAssistant: Using LLM Assistant More Affordably and Accurately</a></div>
    <div class="paper-meta">
      📅 2023-10-03
    </div>
    <details class="paper-abstract">
      Today, users ask Large language models (LLMs) as assistants to answer queries that require external knowledge; they ask about the weather in a specific city, about stock prices, and even about where specific locations are within their neighborhood. These queries require the LLM to produce code that invokes external APIs to answer the user's question, yet LLMs rarely produce correct code on the first try, requiring iterative code refinement upon execution results. In addition, using LLM assistants to support high query volumes can be expensive. In this work, we contribute a framework, EcoAssistant, that enables LLMs to answer code-driven queries more affordably and accurately. EcoAssistant contains three components. First, it allows the LLM assistants to converse with an automatic code executor to iteratively refine code or to produce answers based on the execution results. Second, we use a hierarchy of LLM assistants, which attempts to answer the query with weaker, cheaper LLMs before backing off to stronger, expensive ones. Third, we retrieve solutions from past successful queries as in-context demonstrations to help subsequent queries. Empirically, we show that EcoAssistant offers distinct advantages for affordability and accuracy, surpassing GPT-4 by 10 points of success rate with less than 50% of GPT-4's cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.02431v1">Can Large Language Models Provide Security & Privacy Advice? Measuring the Ability of LLMs to Refute Misconceptions</a></div>
    <div class="paper-meta">
      📅 2023-10-03
      | 💬 Accepted to the Annual Computer Security Applications Conference (ACSAC), 2023
    </div>
    <details class="paper-abstract">
      Users seek security & privacy (S&P) advice from online resources, including trusted websites and content-sharing platforms. These resources help users understand S&P technologies and tools and suggest actionable strategies. Large Language Models (LLMs) have recently emerged as trusted information sources. However, their accuracy and correctness have been called into question. Prior research has outlined the shortcomings of LLMs in answering multiple-choice questions and user ability to inadvertently circumvent model restrictions (e.g., to produce toxic content). Yet, the ability of LLMs to provide reliable S&P advice is not well-explored. In this paper, we measure their ability to refute popular S&P misconceptions that the general public holds. We first study recent academic literature to curate a dataset of over a hundred S&P-related misconceptions across six different topics. We then query two popular LLMs (Bard and ChatGPT) and develop a labeling guide to evaluate their responses to these misconceptions. To comprehensively evaluate their responses, we further apply three strategies: query each misconception multiple times, generate and query their paraphrases, and solicit source URLs of the responses. Both models demonstrate, on average, a 21.3% non-negligible error rate, incorrectly supporting popular S&P misconceptions. The error rate increases to 32.6% when we repeatedly query LLMs with the same or paraphrased misconceptions. We also expose that models may partially support a misconception or remain noncommittal, refusing a firm stance on misconceptions. Our exploration of information sources for responses revealed that LLMs are susceptible to providing invalid URLs (21.2% for Bard and 67.7% for ChatGPT) or point to unrelated sources (44.2% returned by Bard and 18.3% by ChatGPT).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.08155v2">AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation</a></div>
    <div class="paper-meta">
      📅 2023-10-03
      | 💬 43 pages (10 pages for the main text, 3 pages for references, and 30 pages for appendices)
    </div>
    <details class="paper-abstract">
      AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.10490v4">Abusing Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs</a></div>
    <div class="paper-meta">
      📅 2023-10-03
    </div>
    <details class="paper-abstract">
      We demonstrate how images and sounds can be used for indirect prompt and instruction injection in multi-modal LLMs. An attacker generates an adversarial perturbation corresponding to the prompt and blends it into an image or audio recording. When the user asks the (unmodified, benign) model about the perturbed image or audio, the perturbation steers the model to output the attacker-chosen text and/or make the subsequent dialog follow the attacker's instruction. We illustrate this attack with several proof-of-concept examples targeting LLaVa and PandaGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01260v1">SPELL: Semantic Prompt Evolution based on a LLM</a></div>
    <div class="paper-meta">
      📅 2023-10-02
    </div>
    <details class="paper-abstract">
      Prompt engineering is a new paradigm for enhancing the performance of trained neural network models. For optimizing text-style prompts, existing methods usually individually operate small portions of a text step by step, which either breaks the fluency or could not globally adjust a prompt. Since large language models (LLMs) have powerful ability of generating coherent texts token by token, can we utilize LLMs for improving prompts? Based on this motivation, in this paper, considering a trained LLM as a text generator, we attempt to design a black-box evolution algorithm for automatically optimizing texts, namely SPELL (Semantic Prompt Evolution based on a LLM). The proposed method is evaluated with different LLMs and evolution parameters in different text tasks. Experimental results show that SPELL could rapidly improve the prompts indeed. We further explore the evolution process and discuss on the limitations, potential possibilities and future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.00789v1">Testing the Limits of Unified Sequence to Sequence LLM Pretraining on Diverse Table Data Tasks</a></div>
    <div class="paper-meta">
      📅 2023-10-01
    </div>
    <details class="paper-abstract">
      Tables stored in databases and tables which are present in web pages and articles account for a large part of semi-structured data that is available on the internet. It then becomes pertinent to develop a modeling approach with large language models (LLMs) that can be used to solve diverse table tasks such as semantic parsing, question answering as well as classification problems. Traditionally, there existed separate models specialized for each task individually. It raises the question of how far can we go to build a unified model that works well on some table tasks without significant degradation on others. To that end, we attempt at creating a shared modeling approach in the pretraining stage with encoder-decoder style LLMs that can cater to diverse tasks. We evaluate our approach that continually pretrains and finetunes different model families of T5 with data from tables and surrounding context, on these downstream tasks at different model scales. Through multiple ablation studies, we observe that our pretraining with self-supervised objectives can significantly boost the performance of the models on these tasks. As an example of one improvement, we observe that the instruction finetuned public models which come specialized on text question answering (QA) and have been trained on table data still have room for improvement when it comes to table specific QA. Our work is the first attempt at studying the advantages of a unified approach to table specific pretraining when scaled from 770M to 11B sequence to sequence models while also comparing the instruction finetuned variants of the models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.00576v1">GrowLength: Accelerating LLMs Pretraining by Progressively Growing Training Length</a></div>
    <div class="paper-meta">
      📅 2023-10-01
    </div>
    <details class="paper-abstract">
      The evolving sophistication and intricacies of Large Language Models (LLMs) yield unprecedented advancements, yet they simultaneously demand considerable computational resources and incur significant costs. To alleviate these challenges, this paper introduces a novel, simple, and effective method named ``\growlength'' to accelerate the pretraining process of LLMs. Our method progressively increases the training length throughout the pretraining phase, thereby mitigating computational costs and enhancing efficiency. For instance, it begins with a sequence length of 128 and progressively extends to 4096. This approach enables models to process a larger number of tokens within limited time frames, potentially boosting their performance. In other words, the efficiency gain is derived from training with shorter sequences optimizing the utilization of resources. Our extensive experiments with various state-of-the-art LLMs have revealed that models trained using our method not only converge more swiftly but also exhibit superior performance metrics compared to those trained with existing methods. Furthermore, our method for LLMs pretraining acceleration does not require any additional engineering efforts, making it a practical solution in the realm of LLMs.
    </details>
</div>
