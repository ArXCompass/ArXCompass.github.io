# llm - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07870v6">Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Although people are impressed by the content generation skills of large language models, the use of LLMs, such as ChatGPT, is limited by the domain grounding of the content. The correctness and groundedness of the generated content need to be based on a verified context, such as results from Retrieval-Augmented Generation (RAG). One important issue when adapting LLMs to a customized domain is that the generated responses are often incomplete, or the additions are not verified and may even be hallucinated. Prior studies on hallucination detection have focused on evaluation metrics, which are not easily adaptable to dynamic domains and can be vulnerable to attacks like jail-breaking. In this work, we propose 1) a post-processing algorithm that leverages knowledge triplets in RAG context to correct hallucinations and 2) a dual-decoder model that fuses RAG context to guide the generation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.09569v2">Are You Human? An Adversarial Benchmark to Expose LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated an alarming ability to impersonate humans in conversation, raising concerns about their potential misuse in scams and deception. Humans have a right to know if they are conversing to an LLM. We evaluate text-based prompts designed as challenges to expose LLM imposters in real-time. To this end we compile and release an open-source benchmark dataset that includes 'implicit challenges' that exploit an LLM's instruction-following mechanism to cause role deviation, and 'exlicit challenges' that test an LLM's ability to perform simple tasks typically easy for humans but difficult for LLMs. Our evaluation of 9 leading models from the LMSYS leaderboard revealed that explicit challenges successfully detected LLMs in 78.4% of cases, while implicit challenges were effective in 22.9% of instances. User studies validate the real-world applicability of our methods, with humans outperforming LLMs on explicit challenges (78% vs 22% success rate). Our framework unexpectedly revealed that many study participants were using LLMs to complete tasks, demonstrating its effectiveness in detecting both AI impostors and human misuse of AI tools. This work addresses the critical need for reliable, real-time LLM detection methods in high-stakes conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02820v3">DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in complex workflows, where different LLMs and fine-tuned variants collaboratively address complex tasks. However, these systems face significant inefficiencies due to redundant context processing of the shared context. We propose DroidSpeak, a framework that optimizes context sharing between fine-tuned LLMs derived from the same foundational model. DroidSpeak identifies critical layers in the KV cache and selectively recomputes them, enabling effective reuse of intermediate data while maintaining high accuracy. Our approach balances computational efficiency and task fidelity, significantly reducing inference latency and throughput bottlenecks. Experiments on diverse datasets and model pairs demonstrate that DroidSpeak achieves up to 3x higher throughputs and 2.6x faster prefill times with negligible accuracy loss compared to full recomputation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15453v1">Northeastern Uni at Multilingual Counterspeech Generation: Enhancing Counter Speech Generation with LLM Alignment through Direct Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 10 pages, 6 tables, 1 figure, The First Workshop on Multilingual Counterspeech Generation (MCG) at The 31st International Conference on Computational Linguistics (COLING 2025)
    </div>
    <details class="paper-abstract">
      The automatic generation of counter-speech (CS) is a critical strategy for addressing hate speech by providing constructive and informed responses. However, existing methods often fail to generate high-quality, impactful, and scalable CS, particularly across diverse linguistic contexts. In this paper, we propose a novel methodology to enhance CS generation by aligning Large Language Models (LLMs) using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). Our approach leverages DPO to align LLM outputs with human preferences, ensuring contextually appropriate and linguistically adaptable responses. Additionally, we incorporate knowledge grounding to enhance the factual accuracy and relevance of generated CS. Experimental results demonstrate that DPO-aligned models significantly outperform SFT baselines on CS benchmarks while scaling effectively to multiple languages. These findings highlight the potential of preference-based alignment techniques to advance CS generation across varied linguistic settings. The model supervision and alignment is done in English and the same model is used for reporting metrics across other languages like Basque, Italian, and Spanish.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15450v1">Fietje: An open, efficient LLM for Dutch</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      This paper introduces Fietje, a family of small language models (SLMs) specifically designed for the Dutch language. The model is based on Phi 2, an English-centric model of 2.7 billion parameters. Fietje demonstrated competitive results with larger language models upon its release. A core emphasis of this work is transparency and reproducibility: Fietje is fully open-source, with model weights, datasets, training, and evaluation code all publicly accessible. The paper discusses the performance of Fietje and many other models on an extensive evaluation suite of benchmarks on reasoning, sentiment analysis, world knowledge, linguistic acceptability and word sense disambiguation. Evaluation results illustrate the rapid progress in the field of LLMs, where recent small models outperform older, larger models that were fine-tuned for Dutch. This trend signals an exciting future for Dutch language processing, suggesting that even compact LLMs are becoming increasingly capable. Furthermore, ongoing and future efforts to adapt LLMs to Dutch are poised to enhance these models even further, broadening their applicability and accessibility. Fietje is only an intermediate step in improving accessibility to language technology for users of the Dutch language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18626v1">Why Do Large Language Models (LLMs) Struggle to Count Letters?</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved unprecedented performance on many complex tasks, being able, for example, to answer questions on almost any topic. However, they struggle with other simple tasks, such as counting the occurrences of letters in a word, as illustrated by the inability of many LLMs to count the number of "r" letters in "strawberry". Several works have studied this problem and linked it to the tokenization used by LLMs, to the intrinsic limitations of the attention mechanism, or to the lack of character-level training data. In this paper, we conduct an experimental study to evaluate the relations between the LLM errors when counting letters with 1) the frequency of the word and its components in the training dataset and 2) the complexity of the counting operation. We present a comprehensive analysis of the errors of LLMs when counting letter occurrences by evaluating a representative group of models over a large number of words. The results show a number of consistent trends in the models evaluated: 1) models are capable of recognizing the letters but not counting them; 2) the frequency of the word and tokens in the word does not have a significant impact on the LLM errors; 3) there is a positive correlation of letter frequency with errors, more frequent letters tend to have more counting errors, 4) the errors show a strong correlation with the number of letters or tokens in a word and 5) the strongest correlation occurs with the number of letters with counts larger than one, with most models being unable to correctly count words in which letters appear more than twice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12856v5">LLM Processes: Numerical Predictive Distributions Conditioned on Natural Language</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Machine learning practitioners often face significant challenges in formally integrating their prior knowledge and beliefs into predictive models, limiting the potential for nuanced and context-aware analyses. Moreover, the expertise needed to integrate this prior knowledge into probabilistic modeling typically limits the application of these models to specialists. Our goal is to build a regression model that can process numerical data and make probabilistic predictions at arbitrary locations, guided by natural language text which describes a user's prior knowledge. Large Language Models (LLMs) provide a useful starting point for designing such a tool since they 1) provide an interface where users can incorporate expert insights in natural language and 2) provide an opportunity for leveraging latent problem-relevant knowledge encoded in LLMs that users may not have themselves. We start by exploring strategies for eliciting explicit, coherent numerical predictive distributions from LLMs. We examine these joint predictive distributions, which we call LLM Processes, over arbitrarily-many quantities in settings such as forecasting, multi-dimensional regression, black-box optimization, and image modeling. We investigate the practical details of prompting to elicit coherent predictive distributions, and demonstrate their effectiveness at regression. Finally, we demonstrate the ability to usefully incorporate text into numerical predictions, improving predictive performance and giving quantitative structure that reflects qualitative descriptions. This lets us begin to explore the rich, grounded hypothesis space that LLMs implicitly encode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15386v1">Systematic Evaluation of Long-Context LLMs on Financial Concepts</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 Accepted at EMNLP 2024
    </div>
    <details class="paper-abstract">
      Long-context large language models (LC LLMs) promise to increase reliability of LLMs in real-world tasks requiring processing and understanding of long input documents. However, this ability of LC LLMs to reliably utilize their growing context windows remains under investigation. In this work, we evaluate the performance of state-of-the-art GPT-4 suite of LC LLMs in solving a series of progressively challenging tasks, as a function of factors such as context length, task difficulty, and position of key information by creating a real world financial news dataset. Our findings indicate that LC LLMs exhibit brittleness at longer context lengths even for simple tasks, with performance deteriorating sharply as task complexity increases. At longer context lengths, these state-of-the-art models experience catastrophic failures in instruction following resulting in degenerate outputs. Our prompt ablations also reveal unfortunate continued sensitivity to both the placement of the task instruction in the context window as well as minor markdown formatting. Finally, we advocate for more rigorous evaluation of LC LLMs by employing holistic metrics such as F1 (rather than recall) and reporting confidence intervals, thereby ensuring robust and conclusive findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15178v1">HPC-Coder-V2: Studying Code LLMs Across Low-Resource Parallel Languages</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based coding tools have been tremendously successful as software development assistants, yet they are often designed for general purpose programming tasks and perform poorly for more specialized domains such as high performance computing. Creating specialized models and tools for these domains is crucial towards gaining the benefits of LLMs in areas such as HPC. While previous work has explored HPC-specific models, LLMs still struggle to generate parallel code and it is not at all clear what hurdles are still holding back these LLMs and what must be done to overcome them. In this work, we conduct an in-depth study along the many axes of fine-tuning a specialized HPC LLM in order to better understand the challenges. Based on our findings we fine-tune and evaluate a specialized HPC LLM that is shown to be the best performing open-source code LLM for parallel code generation to date.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15177v1">Critical-Questions-of-Thought: Steering LLM reasoning with Argumentative Querying</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Studies have underscored how, regardless of the recent breakthrough and swift advances in AI research, even state-of-the-art Large Language models (LLMs) continue to struggle when performing logical and mathematical reasoning. The results seem to suggest that LLMs still work as (highly advanced) data pattern identifiers, scoring poorly when attempting to generalise and solve reasoning problems the models have never previously seen or that are not close to samples presented in their training data. To address this compelling concern, this paper makes use of the notion of critical questions from the literature on argumentation theory, focusing in particular on Toulmin's model of argumentation. We show that employing these critical questions can improve the reasoning capabilities of LLMs. By probing the rationale behind the models' reasoning process, the LLM can assess whether some logical mistake is occurring and correct it before providing the final reply to the user prompt. The underlying idea is drawn from the gold standard of any valid argumentative procedure: the conclusion is valid if it is entailed by accepted premises. Or, to paraphrase such Aristotelian principle in a real-world approximation, characterised by incomplete information and presumptive logic, the conclusion is valid if not proved otherwise. This approach successfully steers the models' output through a reasoning pipeline, resulting in better performance against the baseline and its Chain-of-Thought (CoT) implementation. To this end, an extensive evaluation of the proposed approach on the MT-Bench Reasoning and Math tasks across a range of LLMs is provided.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06289v3">S$^{2}$FT: Efficient, Scalable and Generalizable LLM Fine-tuning by Structured Sparsity</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Current PEFT methods for LLMs can achieve either high quality, efficient training, or scalable serving, but not all three simultaneously. To address this limitation, we investigate sparse fine-tuning and observe a remarkable improvement in generalization ability. Utilizing this key insight, we propose a family of Structured Sparse Fine-Tuning (S$^{2}$FT) methods for LLMs, which concurrently achieve state-of-the-art fine-tuning performance, training efficiency, and inference scalability. S$^{2}$FT accomplishes this by "selecting sparsely and computing densely". It selects a few heads and channels in the MHA and FFN modules for each Transformer block, respectively. Next, it co-permutes weight matrices on both sides of the coupled structures in LLMs to connect the selected components in each layer into a dense submatrix. Finally, S$^{2}$FT performs in-place gradient updates on all submatrices. Through theoretical analysis and empirical results, our method prevents forgetting while simplifying optimization, delivers SOTA performance on both commonsense and arithmetic reasoning with 4.6% and 1.3% average improvements compared to LoRA, and surpasses full FT by 11.5% when generalizing to various domains after instruction tuning. Using our partial backpropagation algorithm, S$^{2}$FT saves training memory up to 3$\times$ and improves latency by 1.5-2.7$\times$ compared to full FT, while delivering an average 10% improvement over LoRA on both metrics. We further demonstrate that the weight updates in S$^{2}$FT can be decoupled into adapters, enabling effective fusion, fast switch, and efficient parallelism for serving multiple fine-tuned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05199v2">CodeLutra: Boosting LLM Code Generation via Preference-Guided Refinement</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 16 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized code generation but require significant resources and often over-generalize, limiting their task-specific efficiency. Fine-tuning smaller, open-source LLMs provides a cost-effective alternative. However, standard supervised approaches rely only on correct examples, missing valuable insights from failures. We introduce CodeLutra, a framework that leverages both correct and incorrect code attempts. Instead of using only correct solutions, CodeLutra applies iterative preference-based refinement, comparing successful and failed outputs to better approximate desired results. This approach narrows the performance gap with state-of-the-art larger models without requiring massive datasets or auxiliary models. For instance, on a challenging data science coding task, using only 500 samples improved Llama-3-8B's accuracy from 28.2% to 48.6%, approaching GPT-4's level. By learning from both successes and mistakes, CodeLutra provides a scalable and efficient path to high-quality code generation, making smaller open-source models more competitive with leading closed-source alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15156v1">Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Text-to-video models have made remarkable advancements through optimization on high-quality text-video pairs, where the textual prompts play a pivotal role in determining quality of output videos. However, achieving the desired output often entails multiple revisions and iterative inference to refine user-provided prompts. Current automatic methods for refining prompts encounter challenges such as Modality-Inconsistency, Cost-Discrepancy, and Model-Unaware when applied to text-to-video diffusion models. To address these problem, we introduce an LLM-based prompt adaptation framework, termed as Prompt-A-Video, which excels in crafting Video-Centric, Labor-Free and Preference-Aligned prompts tailored to specific video diffusion model. Our approach involves a meticulously crafted two-stage optimization and alignment system. Initially, we conduct a reward-guided prompt evolution pipeline to automatically create optimal prompts pool and leverage them for supervised fine-tuning (SFT) of the LLM. Then multi-dimensional rewards are employed to generate pairwise data for the SFT model, followed by the direct preference optimization (DPO) algorithm to further facilitate preference alignment. Through extensive experimentation and comparative analyses, we validate the effectiveness of Prompt-A-Video across diverse generation models, highlighting its potential to push the boundaries of video generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11917v3">Does VLM Classification Benefit from LLM Description Semantics?</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 AAAI-25 (extended version), Code: https://github.com/CompVis/DisCLIP
    </div>
    <details class="paper-abstract">
      Accurately describing images with text is a foundation of explainable AI. Vision-Language Models (VLMs) like CLIP have recently addressed this by aligning images and texts in a shared embedding space, expressing semantic similarities between vision and language embeddings. VLM classification can be improved with descriptions generated by Large Language Models (LLMs). However, it is difficult to determine the contribution of actual description semantics, as the performance gain may also stem from a semantic-agnostic ensembling effect, where multiple modified text prompts act as a noisy test-time augmentation for the original one. We propose an alternative evaluation scenario to decide if a performance boost of LLM-generated descriptions is caused by such a noise augmentation effect or rather by genuine description semantics. The proposed scenario avoids noisy test-time augmentation and ensures that genuine, distinctive descriptions cause the performance boost. Furthermore, we propose a training-free method for selecting discriminative descriptions that work independently of classname-ensembling effects. Our approach identifies descriptions that effectively differentiate classes within a local CLIP label neighborhood, improving classification accuracy across seven datasets. Additionally, we provide insights into the explainability of description-based image classification with VLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15035v1">LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Gaps</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Building safe Large Language Models (LLMs) across multiple languages is essential in ensuring both safe access and linguistic diversity. To this end, we introduce M-ALERT, a multilingual benchmark that evaluates the safety of LLMs in five languages: English, French, German, Italian, and Spanish. M-ALERT includes 15k high-quality prompts per language, totaling 75k, following the detailed ALERT taxonomy. Our extensive experiments on 10 state-of-the-art LLMs highlight the importance of language-specific safety analysis, revealing that models often exhibit significant inconsistencies in safety across languages and categories. For instance, Llama3.2 shows high unsafety in the category crime_tax for Italian but remains safe in other languages. Similar differences can be observed across all models. In contrast, certain categories, such as substance_cannabis and crime_propaganda, consistently trigger unsafe responses across models and languages. These findings underscore the need for robust multilingual safety practices in LLMs to ensure safe and responsible usage across diverse user communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14512v3">LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Zero-shot graph machine learning, especially with graph neural networks (GNNs), has garnered significant interest due to the challenge of scarce labeled data. While methods like self-supervised learning and graph prompt learning have been extensively explored, they often rely on fine-tuning with task-specific labels, limiting their effectiveness in zero-shot scenarios. Inspired by the zero-shot capabilities of instruction-fine-tuned large language models (LLMs), we introduce a novel framework named Token Embedding-Aligned Graph Language Model (TEA-GLM) that leverages LLMs as cross-dataset and cross-task zero-shot learners for graph machine learning. Concretely, we pretrain a GNN, aligning its representations with token embeddings of an LLM. We then train a linear projector that transforms the GNN's representations into a fixed number of graph token embeddings without tuning the LLM. A unified instruction is designed for various graph tasks at different levels, such as node classification (node-level) and link prediction (edge-level). These design choices collectively enhance our method's effectiveness in zero-shot learning, setting it apart from existing methods. Experiments show that our graph token embeddings help the LLM predictor achieve state-of-the-art performance on unseen datasets and tasks compared to other methods using LLMs as predictors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14995v1">HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 18 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Automatic Heuristic Design (AHD) is an active research area due to its utility in solving complex search and NP-hard combinatorial optimization problems in the real world. The recent advancements in Large Language Models (LLMs) introduce new possibilities by coupling LLMs with evolutionary computation to automatically generate heuristics, known as LLM-based Evolutionary Program Search (LLM-EPS). While previous LLM-EPS studies obtained great performance on various tasks, there is still a gap in understanding the properties of heuristic search spaces and achieving a balance between exploration and exploitation, which is a critical factor in large heuristic search spaces. In this study, we address this gap by proposing two diversity measurement metrics and perform an analysis on previous LLM-EPS approaches, including FunSearch, EoH, and ReEvo. Results on black-box AHD problems reveal that while EoH demonstrates higher diversity than FunSearch and ReEvo, its objective score is unstable. Conversely, ReEvo's reflection mechanism yields good objective scores but fails to optimize diversity effectively. With this finding in mind, we introduce HSEvo, an adaptive LLM-EPS framework that maintains a balance between diversity and convergence with a harmony search algorithm. Through experimentation, we find that HSEvo achieved high diversity indices and good objective scores while remaining cost-effective. These results underscore the importance of balancing exploration and exploitation and understanding heuristic search spaces in designing frameworks in LLM-EPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13765v2">LLM-SEM: A Sentiment-Based Student Engagement Metric Using LLMS for E-Learning Platforms</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Current methods for analyzing student engagement in e-learning platforms, including automated systems, often struggle with challenges such as handling fuzzy sentiment in text comments and relying on limited metadata. Traditional approaches, such as surveys and questionnaires, also face issues like small sample sizes and scalability. In this paper, we introduce LLM-SEM (Language Model-Based Student Engagement Metric), a novel approach that leverages video metadata and sentiment analysis of student comments to measure engagement. By utilizing recent Large Language Models (LLMs), we generate high-quality sentiment predictions to mitigate text fuzziness and normalize key features such as views and likes. Our holistic method combines comprehensive metadata with sentiment polarity scores to gauge engagement at both the course and lesson levels. Extensive experiments were conducted to evaluate various LLM models, demonstrating the effectiveness of LLM-SEM in providing a scalable and accurate measure of student engagement. We fine-tuned TXLM-RoBERTa using human-annotated sentiment datasets to enhance prediction accuracy and utilized LLama 3B, and Gemma 9B from Ollama.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14959v1">Understanding the Dark Side of LLMs' Intrinsic Self-Correction</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction was proposed to improve LLMs' responses via feedback prompts solely based on their inherent capability. However, recent works show that LLMs' intrinsic self-correction fails without oracle labels as feedback prompts. In this paper, we aim to interpret LLMs' intrinsic self-correction for different tasks, especially for those failure cases. By including one simple task and three complex tasks with state-of-the-art (SOTA) LLMs like ChatGPT families (o1, 4o, 3.5-turbo) and Llama families (2-7B, 3-8B, and 3.1-8B), we design three interpretation methods to reveal the dark side of LLMs' intrinsic self-correction. We identify intrinsic self-correction can (1) cause LLMs to waver both intermedia and final answers and lead to prompt bias on simple factual questions; (2) introduce human-like cognitive bias on complex tasks. In light of our findings, we also provide two simple yet effective strategies for alleviation: question repeating and supervised fine-tuning with a few samples. We open-source our work at https://x-isc.info/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07991v4">Human and LLM Biases in Hate Speech Annotations: A Socio-Demographic Analysis of Annotators and Targets</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      The rise of online platforms exacerbated the spread of hate speech, demanding scalable and effective detection. However, the accuracy of hate speech detection systems heavily relies on human-labeled data, which is inherently susceptible to biases. While previous work has examined the issue, the interplay between the characteristics of the annotator and those of the target of the hate are still unexplored. We fill this gap by leveraging an extensive dataset with rich socio-demographic information of both annotators and targets, uncovering how human biases manifest in relation to the target's attributes. Our analysis surfaces the presence of widespread biases, which we quantitatively describe and characterize based on their intensity and prevalence, revealing marked differences. Furthermore, we compare human biases with those exhibited by persona-based LLMs. Our findings indicate that while persona-based LLMs do exhibit biases, these differ significantly from those of human annotators. Overall, our work offers new and nuanced results on human biases in hate speech annotations, as well as fresh insights into the design of AI-driven hate speech detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00467v3">Dynamic Planning for LLM-based Graphical User Interface Automation</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has spurred considerable interest in advancing autonomous LLMs-based agents, particularly in intriguing applications within smartphone graphical user interfaces (GUIs). When presented with a task goal, these agents typically emulate human actions within a GUI environment until the task is completed. However, a key challenge lies in devising effective plans to guide action prediction in GUI tasks, though planning have been widely recognized as effective for decomposing complex tasks into a series of steps. Specifically, given the dynamic nature of environmental GUIs following action execution, it is crucial to dynamically adapt plans based on environmental feedback and action history.We show that the widely-used ReAct approach fails due to the excessively long historical dialogues. To address this challenge, we propose a novel approach called Dynamic Planning of Thoughts (D-PoT) for LLM-based GUI agents.D-PoT involves the dynamic adjustment of planning based on the environmental feedback and execution history. Experimental results reveal that the proposed D-PoT significantly surpassed the strong GPT-4V baseline by +12.7% (34.66% $\rightarrow$ 47.36%) in accuracy. The analysis highlights the generality of dynamic planning in different backbone LLMs, as well as the benefits in mitigating hallucinations and adapting to unseen tasks. Code is available at https://github.com/sqzhang-lazy/D-PoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14838v1">DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Efficient KV cache management in LLMs is crucial for long-context tasks like RAG and summarization. Existing KV cache compression methods enforce a fixed pattern, neglecting task-specific characteristics and reducing the retention of essential information. However, we observe distinct activation patterns across layers in various tasks, highlighting the need for adaptive strategies tailored to each task's unique demands. Based on this insight, we propose DynamicKV, a method that dynamically optimizes token retention by adjusting the number of tokens retained at each layer to adapt to the specific task. DynamicKV establishes global and per-layer maximum KV cache budgets, temporarily retaining the maximum budget for the current layer, and periodically updating the KV cache sizes of all preceding layers during inference. Our method retains only 1.7% of the KV cache size while achieving ~85% of the Full KV cache performance on LongBench. Notably, even under extreme compression (0.9%), DynamicKV surpasses state-of-the-art (SOTA) methods by 11% in the Needle-in-a-Haystack test using Mistral-7B-Instruct-v0.2. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14837v1">ObjVariantEnsemble: Advancing Point Cloud LLM Evaluation in Challenging Scenes with Subtly Distinguished Objects</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 Accepted to AAAI2025
    </div>
    <details class="paper-abstract">
      3D scene understanding is an important task, and there has been a recent surge of research interest in aligning 3D representations of point clouds with text to empower embodied AI. However, due to the lack of comprehensive 3D benchmarks, the capabilities of 3D models in real-world scenes, particularly those that are challenging with subtly distinguished objects, remain insufficiently investigated. To facilitate a more thorough evaluation of 3D models' capabilities, we propose a scheme, ObjVariantEnsemble, to systematically introduce more scenes with specified object classes, colors, shapes, quantities, and spatial relationships to meet model evaluation needs. More importantly, we intentionally construct scenes with similar objects to a certain degree and design an LLM-VLM-cooperated annotator to capture key distinctions as annotations. The resultant benchmark can better challenge 3D models, reveal their shortcomings in understanding, and potentially aid in the further development of 3D models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14771v1">ALKAFI-LLAMA3: Fine-Tuning LLMs for Precise Legal Understanding in Palestine</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in diverse domains, yet their application in the legal sector, particularly in low-resource contexts, remains limited. This study addresses the challenges of adapting LLMs to the Palestinian legal domain, where political instability, fragmented legal frameworks, and limited AI resources hinder effective machine-learning applications. We present a fine-tuned model based on a quantized version of Llama-3.2-1B-Instruct, trained on a synthetic data set derived from Palestinian legal texts. Using smaller-scale models and strategically generated question-answer pairs, we achieve a cost-effective, locally sustainable solution that provides accurate and contextually relevant legal guidance. Our experiments demonstrate promising performance on various query types, ranging from yes/no questions and narrative explanations to complex legal differentiations, while highlighting areas for improvement, such as handling calculation-based inquiries and structured list formatting. This work provides a pathway for the deployment of AI-driven legal assistance tools tailored to the needs of resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14737v1">On Verbalized Confidence Scores for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) and their tight integration into our daily life make it essential to dedicate efforts towards their trustworthiness. Uncertainty quantification for LLMs can establish more human trust into their responses, but also allows LLM agents to make more informed decisions based on each other's uncertainty. To estimate the uncertainty in a response, internal token logits, task-specific proxy models, or sampling of multiple responses are commonly used. This work focuses on asking the LLM itself to verbalize its uncertainty with a confidence score as part of its output tokens, which is a promising way for prompt- and model-agnostic uncertainty quantification with low overhead. Using an extensive benchmark, we assess the reliability of verbalized confidence scores with respect to different datasets, models, and prompt methods. Our results reveal that the reliability of these scores strongly depends on how the model is asked, but also that it is possible to extract well-calibrated confidence scores with certain prompt methods. We argue that verbalized confidence scores can become a simple but effective and versatile uncertainty quantification method in the future. Our code is available at https://github.com/danielyxyang/llm-verbalized-uq .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18403v2">LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      There is an increasing trend towards evaluating NLP models with LLMs instead of human judgments, raising questions about the validity of these evaluations, as well as their reproducibility in the case of proprietary models. We provide JUDGE-BENCH, an extensible collection of 20 NLP datasets with human annotations covering a broad range of evaluated properties and types of data, and comprehensively evaluate 11 current LLMs, covering both open-weight and proprietary models, for their ability to replicate the annotations. Our evaluations show substantial variance across models and datasets. Models are reliable evaluators on some tasks, but overall display substantial variability depending on the property being evaluated, the expertise level of the human judges, and whether the language is human or model-generated. We conclude that LLMs should be carefully validated against human judgments before being used as evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11242v2">TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Specializing large language models (LLMs) for local deployment in domain-specific use cases is necessary for strong performance while meeting latency and privacy constraints. However, conventional task-specific adaptation approaches do not show simultaneous memory saving and inference speedup at deployment time. Practical compression techniques like quantization and pruning require dedicated hardware or kernel support to achieve measured inference speedup. We develop TrimLLM based on the layer-wise specialization phenomenon we empirically observed and verified on contemporary LLMs. TrimLLM reduces the depth of LLMs via progressive layer dropping. We show it retains LLMs' capacity in specific domains and achieves inference speedup irrespective of hardware and deep learning frameworks. We evaluated TrimLLM on LLMs of various sizes for inference; models adapted on medical, legal, and financial datasets all demonstrate $2.1-5.7\times$ inference speedup on consumer GPUs and up to $3.1\times$ speedup on A100 when compared to state-of-the-art model compression algorithms, with no loss in accuracy at 50$\sim$60\% model compression ratio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14675v1">LLMs as mediators: Can they diagnose conflicts accurately?</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 27 pages, 2 appendices, 21 tables (incl appendices)
    </div>
    <details class="paper-abstract">
      Prior research indicates that to be able to mediate conflict, observers of disagreements between parties must be able to reliably distinguish the sources of their disagreement as stemming from differences in beliefs about what is true (causality) vs. differences in what they value (morality). In this paper, we test if OpenAI's Large Language Models GPT 3.5 and GPT 4 can perform this task and whether one or other type of disagreement proves particularly challenging for LLM's to diagnose. We replicate study 1 in Ko\c{c}ak et al. (2003), which employes a vignette design, with OpenAI's GPT 3.5 and GPT 4. We find that both LLMs have similar semantic understanding of the distinction between causal and moral codes as humans and can reliably distinguish between them. When asked to diagnose the source of disagreement in a conversation, both LLMs, compared to humans, exhibit a tendency to overestimate the extent of causal disagreement and underestimate the extent of moral disagreement in the moral misalignment condition. This tendency is especially pronounced for GPT 4 when using a proximate scale that relies on concrete language specific to an issue. GPT 3.5 does not perform as well as GPT4 or humans when using either the proximate or the distal scale. The study provides a first test of the potential for using LLMs to mediate conflict by diagnosing the root of disagreements in causal and evaluative codes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14656v1">Length Controlled Generation for Black-box LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive instruction following capabilities, while still struggling to accurately manage the length of the generated text, which is a fundamental requirement in many real-world applications. Existing length control methods involve fine-tuning the parameters of LLMs, which is inefficient and suboptimal for practical use. In this paper, we propose a novel iterative sampling framework for text length control, integrating the Metropolis-Hastings algorithm with an importance sampling acceleration strategy. This framework efficiently and reliably regulates LLMs to generate length-constrained text without modifying the underlying parameters, thereby preserving the original capabilities of LLMs. Experimental results demonstrate that our framework achieves almost 100\% success rates of length control on Llama3.1 for tasks such as length-controlled abstractive summarization and length-constrained instruction following, with minimal additional computational overhead. This also highlights the significant potential of our method for precise length control across a broader range of applications, without compromising the versatility of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14642v1">TOMG-Bench: Evaluating LLMs on Text-based Open Molecule Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 A benchmark for text-based open molecule generation
    </div>
    <details class="paper-abstract">
      In this paper, we propose Text-based Open Molecule Generation Benchmark (TOMG-Bench), the first benchmark to evaluate the open-domain molecule generation capability of LLMs. TOMG-Bench encompasses a dataset of three major tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom). Each task further contains three subtasks, with each subtask comprising 5,000 test samples. Given the inherent complexity of open molecule generation, we have also developed an automated evaluation system that helps measure both the quality and the accuracy of the generated molecules. Our comprehensive benchmarking of 25 LLMs reveals the current limitations and potential areas for improvement in text-guided molecule discovery. Furthermore, with the assistance of OpenMolIns, a specialized instruction tuning dataset proposed for solving challenges raised by TOMG-Bench, Llama3.1-8B could outperform all the open-source general LLMs, even surpassing GPT-3.5-turbo by 46.5\% on TOMG-Bench. Our codes and datasets are available through https://github.com/phenixace/TOMG-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14611v1">Is This You, LLM? Recognizing AI-written Programs with Multilingual Code Stylometry</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      With the increasing popularity of LLM-based code completers, like GitHub Copilot, the interest in automatically detecting AI-generated code is also increasing-in particular in contexts where the use of LLMs to program is forbidden by policy due to security, intellectual property, or ethical concerns.We introduce a novel technique for AI code stylometry, i.e., the ability to distinguish code generated by LLMs from code written by humans, based on a transformer-based encoder classifier. Differently from previous work, our classifier is capable of detecting AI-written code across 10 different programming languages with a single machine learning model, maintaining high average accuracy across all languages (84.1% $\pm$ 3.8%).Together with the classifier we also release H-AIRosettaMP, a novel open dataset for AI code stylometry tasks, consisting of 121 247 code snippets in 10 popular programming languages, labeled as either human-written or AI-generated. The experimental pipeline (dataset, training code, resulting models) is the first fully reproducible one for the AI code stylometry task. Most notably our experiments rely only on open LLMs, rather than on proprietary/closed ones like ChatGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14590v1">MixLLM: LLM Quantization with Global Mixed-precision between Output-features and Highly-efficient System Design</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 The code will be released in the future
    </div>
    <details class="paper-abstract">
      Quantization has become one of the most effective methodologies to compress LLMs into smaller size. However, the existing quantization solutions still show limitations of either non-negligible accuracy drop or system inefficiency. In this paper, we make a comprehensive analysis of the general quantization principles on their effect to the triangle of accuracy, memory consumption and system efficiency. We propose MixLLM that explores the new optimization space of mixed-precision quantization between output features based on the insight that different output features matter differently in the model. MixLLM identifies the output features with high salience in the global view rather than within each single layer, effectively assigning the larger bit-width to output features that need it most to achieve good accuracy with low memory consumption. We present the sweet spot of quantization configuration of algorithm-system co-design that leads to high accuracy and system efficiency. To address the system challenge, we design the two-step dequantization to make use of the int8 Tensor Core easily and fast data type conversion to reduce dequantization overhead significantly, and present the software pipeline to overlap the memory access, dequantization and the MatMul to the best. Extensive experiments show that with only 10% more bits, the PPL increasement can be reduced from about 0.5 in SOTA to within 0.2 for Llama 3.1 70B, while on average MMLU-Pro improves by 0.93 over the SOTA of three popular models. In addition to its superior accuracy, MixLLM also achieves state-of-the-art system efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14556v1">CitaLaw: Enhancing LLM with Citations in Legal Domain</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      In this paper, we propose CitaLaw, the first benchmark designed to evaluate LLMs' ability to produce legally sound responses with appropriate citations. CitaLaw features a diverse set of legal questions for both laypersons and practitioners, paired with a comprehensive corpus of law articles and precedent cases as a reference pool. This framework enables LLM-based systems to retrieve supporting citations from the reference corpus and align these citations with the corresponding sentences in their responses. Moreover, we introduce syllogism-inspired evaluation methods to assess the legal alignment between retrieved references and LLM-generated responses, as well as their consistency with user questions. Extensive experiments on 2 open-domain and 7 legal-specific LLMs demonstrate that integrating legal references substantially enhances response quality. Furthermore, our proposed syllogism-based evaluation method exhibits strong agreement with human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15289v1">SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14501v1">Do Large Language Models Defend Inferentialist Semantics?: On the Logical Expressivism and Anti-Representationalism of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      The philosophy of language, which has historically been developed through an anthropocentric lens, is now being forced to move towards post-anthropocentrism due to the advent of large language models (LLMs) like ChatGPT (OpenAI), Claude (Anthropic), which are considered to possess linguistic abilities comparable to those of humans. Traditionally, LLMs have been explained through distributional semantics as their foundational semantics. However, recent research is exploring alternative foundational semantics beyond distributional semantics. This paper proposes Robert Brandom's inferentialist semantics as an suitable foundational semantics for LLMs, specifically focusing on the issue of linguistic representationalism within this post-anthropocentric trend. Here, we show that the anti-representationalism and logical expressivism of inferential semantics, as well as quasi-compositionality, are useful in interpreting the characteristics and behaviors of LLMs. Further, we propose a \emph{consensus theory of truths} for LLMs. This paper argues that the characteristics of LLMs challenge mainstream assumptions in philosophy of language, such as semantic externalism and compositionality. We believe the argument in this paper leads to a re-evaluation of anti\hyphen{}representationalist views of language, potentially leading to new developments in the philosophy of language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09529v2">Can Modern LLMs Act as Agent Cores in Radiology Environments?</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 22 pages,7 figures
    </div>
    <details class="paper-abstract">
      Advancements in large language models (LLMs) have paved the way for LLM-based agent systems that offer enhanced accuracy and interpretability across various domains. Radiology, with its complex analytical requirements, is an ideal field for the application of these agents. This paper aims to investigate the pre-requisite question for building concrete radiology agents which is, `Can modern LLMs act as agent cores in radiology environments?' To investigate it, we introduce RadABench with three-fold contributions: First, we present RadABench-Data, a comprehensive synthetic evaluation dataset for LLM-based agents, generated from an extensive taxonomy encompassing 6 anatomies, 5 imaging modalities, 10 tool categories, and 11 radiology tasks. Second, we propose RadABench-EvalPlat, a novel evaluation platform for agents featuring a prompt-driven workflow and the capability to simulate a wide range of radiology toolsets. Third, we assess the performance of 7 leading LLMs on our benchmark from 5 perspectives with multiple metrics. Our findings indicate that while current LLMs demonstrate strong capabilities in many areas, they are still not sufficiently advanced to serve as the central agent core in a fully operational radiology agent system. Additionally, we identify key factors influencing the performance of LLM-based agent cores, offering insights for clinicians on how to apply agent systems in real-world radiology practices effectively. All of our code and data are open-sourced in https://github.com/MAGIC-AI4Med/RadABench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14479v1">Frenzy: A Memory-Aware Serverless LLM Training System for Heterogeneous GPU Clusters</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Existing work only effective on a given number of GPUs, often neglecting the complexities involved in manually determining the specific types and quantities of GPUs needed, which can be a significant burden for developers. To address this issue, we propose Frenzy, a memory-aware serverless computing method for heterogeneous GPU clusters. Frenzy allows users to submit models without worrying about underlying hardware resources. First, Frenzy predicts the required number and type of GPUs by estimating the GPU memory usage of the LLM. Then, it employs a low-overhead heterogeneity-aware scheduling method to optimize training efficiency. We validated Frenzy's performance by conducting multi-task LLM training tests on a heterogeneous GPU cluster with three different GPU types. The results show that Frenzy's memory usage prediction accuracy exceeds 92\%, the scheduling overhead is reduced by 10 times, and it reduces the average job completion time by 12\% to 18\% compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14471v1">Why We Build Local Large Language Models: An Observational Analysis from 35 Japanese and Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Why do we build local large language models (LLMs)? What should a local LLM learn from the target language? Which abilities can be transferred from other languages? Do language-specific scaling laws exist? To explore these research questions, we evaluated 35 Japanese, English, and multilingual LLMs on 19 evaluation benchmarks for Japanese and English, taking Japanese as a local language. Adopting an observational approach, we analyzed correlations of benchmark scores, and conducted principal component analysis (PCA) on the scores to derive \textit{ability factors} of local LLMs. We found that training on English text can improve the scores of academic subjects in Japanese (JMMLU). In addition, it is unnecessary to specifically train on Japanese text to enhance abilities for solving Japanese code generation, arithmetic reasoning, commonsense, and reading comprehension tasks. In contrast, training on Japanese text could improve question-answering tasks about Japanese knowledge and English-Japanese translation, which indicates that abilities for solving these two tasks can be regarded as \textit{Japanese abilities} for LLMs. Furthermore, we confirmed that the Japanese abilities scale with the computational budget for Japanese text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14470v1">Agent-SafetyBench: Evaluating the Safety of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-19
      | 💬 23 pages, 9 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed as agents, their integration into interactive environments and tool use introduce new safety challenges beyond those associated with the models themselves. However, the absence of comprehensive benchmarks for evaluating agent safety presents a significant barrier to effective assessment and further improvement. In this paper, we introduce Agent-SafetyBench, a comprehensive benchmark designed to evaluate the safety of LLM agents. Agent-SafetyBench encompasses 349 interaction environments and 2,000 test cases, evaluating 8 categories of safety risks and covering 10 common failure modes frequently encountered in unsafe interactions. Our evaluation of 16 popular LLM agents reveals a concerning result: none of the agents achieves a safety score above 60%. This highlights significant safety challenges in LLM agents and underscores the considerable need for improvement. Through quantitative analysis, we identify critical failure modes and summarize two fundamental safety detects in current LLM agents: lack of robustness and lack of risk awareness. Furthermore, our findings suggest that reliance on defense prompts alone is insufficient to address these safety issues, emphasizing the need for more advanced and robust strategies. We release Agent-SafetyBench at \url{https://github.com/thu-coai/Agent-SafetyBench} to facilitate further research and innovation in agent safety evaluation and improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14461v1">From Human Annotation to LLMs: SILICON Annotation Workflow for Management Research</a></div>
    <div class="paper-meta">
      📅 2024-12-19
    </div>
    <details class="paper-abstract">
      Unstructured text data annotation and analysis are fundamental to management research, often relying on human annotators through crowdsourcing platforms. While Large Language Models (LLMs) promise to provide a cost-effective and efficient alternative to human annotation, there lacks a systematic workflow that evaluate when LLMs are suitable or how to proceed with LLM-based text annotation in a reproducible manner. This paper addresses this methodological gap by introducing the ``SILICON" (\textbf{S}ystematic \textbf{I}nference with \textbf{L}LMs for \textbf{I}nformation \textbf{C}lassificati\textbf{o}n and \textbf{N}otation) workflow. The workflow integrates established principles of human annotation with systematic prompt optimization and model selection, addressing challenges such as developing robust annotation guidelines, establishing high-quality human baselines, optimizing prompts, and ensuring reproducibility across LLMs. We validate the SILICON workflow through seven case studies covering common management research tasks, including business proposal evaluation, dialog intent and breakdown analysis, review attribute detection. Our findings highlight the importance of validating annotation guideline agreement, the superiority of expert-developed human baselines over crowdsourced ones, the iterative nature of prompt optimization, and the necessity of testing multiple LLMs. Notably, we propose a regression-based methodology to empirically compare LLM outputs across prompts and models. Our workflow advances management research by establishing reproducible processes for LLM-based annotation that maintain scientific rigor. We provide practical guidance for researchers to effectively navigate the evolving landscape of generative AI tools effectively while maintaining transparency and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14405v1">ChainRank-DPO: Chain Rank Direct Preference Optimization for LLM Rankers</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable effectiveness in text reranking through works like RankGPT, leveraging their human-like reasoning about relevance. However, supervised fine-tuning for ranking often diminishes these models' general-purpose capabilities, including the crucial reasoning abilities that make them valuable for ranking. We introduce a novel approach integrating Chain-of-Thought prompting with an SFT-DPO (Supervised Fine-Tuning followed by Direct Preference Optimization) pipeline to preserve these capabilities while improving ranking performance. Our experiments on TREC 2019 and 2020 Deep Learning datasets show that our approach outperforms the state-of-the-art RankZephyr while maintaining strong performance on the Massive Multitask Language Understanding (MMLU) benchmark, demonstrating effective preservation of general-purpose capabilities through thoughtful fine-tuning strategies. Our code and data will be publicly released upon the acceptance of the paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01227v3">Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Accepted in AAAI Conference on Artificial Intelligence (AAAI-25)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have triggered a new stream of research focusing on compressing the context length to reduce the computational cost while ensuring the retention of helpful information for LLMs to answer the given question. Token-based removal methods are one of the most prominent approaches in this direction, but risk losing the semantics of the context caused by intermediate token removal, especially under high compression ratios, while also facing challenges in computational efficiency. In this work, we propose context-aware prompt compression (CPC), a sentence-level prompt compression technique where its key innovation is a novel context-aware sentence encoder that provides a relevance score for each sentence for a given question. To train this encoder, we generate a new dataset consisting of questions, positives, and negative pairs where positives are sentences relevant to the question, while negatives are irrelevant context sentences. We train the encoder in a contrastive setup to learn context-aware sentence representations. Our method considerably outperforms prior works on prompt compression on benchmark datasets and is up to 10.93x faster at inference compared to the best token-level compression method. We also find better improvement for shorter length constraints in most benchmarks, showing the effectiveness of our proposed solution in the compression of relevant information in a shorter context. Finally, we release the code and the dataset for quick reproducibility and further development: https://github.com/Workday/cpc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14352v1">A Survey on LLM Inference-Time Self-Improvement</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 The first two authors contribute equally
    </div>
    <details class="paper-abstract">
      Techniques that enhance inference through increased computation at test-time have recently gained attention. In this survey, we investigate the current state of LLM Inference-Time Self-Improvement from three different perspectives: Independent Self-improvement, focusing on enhancements via decoding or sampling methods; Context-Aware Self-Improvement, leveraging additional context or datastore; and Model-Aided Self-Improvement, achieving improvement through model collaboration. We provide a comprehensive review of recent relevant studies, contribute an in-depth taxonomy, and discuss challenges and limitations, offering insights for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17846v1">Enhancing Knowledge Distillation for LLMs with Response-Priming Prompting</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Accepted to SoCal NLP Symposium 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing (NLP) tasks. However, these models are often difficult to deploy due to significant computational requirements and resource constraints. Knowledge distillation (KD) is an effective technique for transferring the performance of larger LLMs to smaller models. Traditional KD methods primarily focus on the direct output of the teacher model, with little emphasis on the role of prompting during knowledge transfer. In this paper, we propose a set of novel response-priming prompting strategies applied in the knowledge distillation pipeline to enhance the performance of student models. Our approach fine-tunes a smaller Llama 3.1 8B Instruct model by distilling knowledge from a quantized Llama 3.1 405B Instruct teacher model. We apply LoRA optimization and evaluate on the GSM8K benchmark. Experimental results demonstrate that integrating reasoning-eliciting prompting into the proposed KD pipeline significantly improves student model performance, offering an efficient way to deploy powerful models in resource-constrained environments. We find that Ground Truth prompting results in a 55\% performance increase on GSM8K for a distilled Llama 3.1 8B Instruct compared to the same model distilled without prompting. A thorough investigation into the self-attention layers of the student models indicates that the more successful prompted models tend to exhibit certain positive behaviors inside their attention heads which can be tied to their increased accuracy. Our implementation can be found at https://github.com/alonso130r/knowledge-distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03525v3">UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      This paper introduces UnSeenTimeQA, a novel data contamination-free time-sensitive question-answering (TSQA) benchmark. It differs from existing TSQA benchmarks by avoiding web-searchable queries grounded in the real-world. We present a series of time-sensitive event scenarios based on synthetically generated facts. It requires large language models (LLMs) to engage in genuine temporal reasoning without depending on the factual knowledge acquired during the pre-training phase. We designed three types of time-sensitive questions to test LLMs' temporal reasoning abilities over sequential and parallel event occurrences. Our evaluation of five LLMs on synthetic fact-based TSQA reveals mixed results: while they perform well on simpler subsets, their overall performance remains inferior as compared to real-world fact-based TSQA. Error analysis of LLM-generated reasoning chains indicates that LLMs face difficulties in reasoning over long-range event dependencies and parallel event timelines that unfold concurrently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14304v1">Multi-OphthaLingua: A Multilingual Benchmark for Assessing and Debiasing LLM Ophthalmological QA in LMICs</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Accepted at the AAAI 2025 Artificial Intelligence for Social Impact Track (AAAI-AISI 2025)
    </div>
    <details class="paper-abstract">
      Current ophthalmology clinical workflows are plagued by over-referrals, long waits, and complex and heterogeneous medical records. Large language models (LLMs) present a promising solution to automate various procedures such as triaging, preliminary tests like visual acuity assessment, and report summaries. However, LLMs have demonstrated significantly varied performance across different languages in natural language question-answering tasks, potentially exacerbating healthcare disparities in Low and Middle-Income Countries (LMICs). This study introduces the first multilingual ophthalmological question-answering benchmark with manually curated questions parallel across languages, allowing for direct cross-lingual comparisons. Our evaluation of 6 popular LLMs across 7 different languages reveals substantial bias across different languages, highlighting risks for clinical deployment of LLMs in LMICs. Existing debiasing methods such as Translation Chain-of-Thought or Retrieval-augmented generation (RAG) by themselves fall short of closing this performance gap, often failing to improve performance across all languages and lacking specificity for the medical domain. To address this issue, We propose CLARA (Cross-Lingual Reflective Agentic system), a novel inference time de-biasing method leveraging retrieval augmented generation and self-verification. Our approach not only improves performance across all languages but also significantly reduces the multilingual bias gap, facilitating equitable LLM application across the globe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14161v1">TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at helping to accelerate or even autonomously perform work-related tasks? The answer to this question has important implications for both industry looking to adopt AI into their workflows, and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper, we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that with the most competitive agent, 24% of the tasks can be completed autonomously. This paints a nuanced picture on task automation with LM agents -- in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14141v1">LLMs can realize combinatorial creativity: generating creative ideas via LLMs for scientific research</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Scientific idea generation has been extensively studied in creativity theory and computational creativity research, providing valuable frameworks for understanding and implementing creative processes. However, recent work using Large Language Models (LLMs) for research idea generation often overlooks these theoretical foundations. We present a framework that explicitly implements combinatorial creativity theory using LLMs, featuring a generalization-level retrieval system for cross-domain knowledge discovery and a structured combinatorial process for idea generation. The retrieval system maps concepts across different abstraction levels to enable meaningful connections between disparate domains, while the combinatorial process systematically analyzes and recombines components to generate novel solutions. Experiments on the OAG-Bench dataset demonstrate our framework's effectiveness, consistently outperforming baseline approaches in generating ideas that align with real research developments (improving similarity scores by 7\%-10\% across multiple metrics). Our results provide strong evidence that LLMs can effectively realize combinatorial creativity when guided by appropriate theoretical frameworks, contributing both to practical advancement of AI-assisted research and theoretical understanding of machine creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14137v1">Design choices made by LLM-based test generators prevent them from finding bugs</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      There is an increasing amount of research and commercial tools for automated test case generation using Large Language Models (LLMs). This paper critically examines whether recent LLM-based test generation tools, such as Codium CoverAgent and CoverUp, can effectively find bugs or unintentionally validate faulty code. Considering bugs are only exposed by failing test cases, we explore the question: can these tools truly achieve the intended objectives of software testing when their test oracles are designed to pass? Using real human-written buggy code as input, we evaluate these tools, showing how LLM-generated tests can fail to detect bugs and, more alarmingly, how their design can worsen the situation by validating bugs in the generated test suite and rejecting bug-revealing tests. These findings raise important questions about the validity of the design behind LLM-based test generation tools and their impact on software quality and test suite reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16179v4">MagicPIG: LSH Sampling for Efficient LLM Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with long context windows have gained significant attention. However, the KV cache, stored to avoid re-computation, becomes a bottleneck. Various dynamic sparse or TopK-based attention approximation methods have been proposed to leverage the common insight that attention is sparse. In this paper, we first show that TopK attention itself suffers from quality degradation in certain downstream tasks because attention is not always as sparse as expected. Rather than selecting the keys and values with the highest attention scores, sampling with theoretical guarantees can provide a better estimation for attention output. To make the sampling-based approximation practical in LLM generation, we propose MagicPIG, a heterogeneous system based on Locality Sensitive Hashing (LSH). MagicPIG significantly reduces the workload of attention computation while preserving high accuracy for diverse tasks. MagicPIG stores the LSH hash tables and runs the attention computation on the CPU, which allows it to serve longer contexts and larger batch sizes with high approximation accuracy. MagicPIG can improve decoding throughput by up to $5\times$ across various GPU hardware and achieve 54ms decoding latency on a single RTX 4090 for Llama-3.1-8B-Instruct model with a context of 96k tokens. The code is available at https://github.com/Infini-AI-Lab/MagicPIG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22159v3">Training LLMs for Generating IEC 61131-3 Structured Text with Online Feedback</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Accepted at LLM4Code Workshop @ ICSE 2025. This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      IEC 61131-3 Structured Text (ST) is a widely used programming language for programmable logic controllers (PLCs) in automation systems. However, generating ST code with LLMs poses unique challenges due to limited data in public training datasets and the complexity of ST language syntax. This paper proposes an approach to fine-tune LLMs for the generation of ST code that leverages a preference-based learning method through an online process involving compiler feedback and evaluation from an LLM-based ST expert. In this framework, the model is iteratively refined and generates new training samples, which are subsequently evaluated by a compiler for syntactical correctness and by a specialized LLM that excels at assessing semantic accuracy, though it is not optimized for code generation itself. This approach results in marked improvements for the trained LLM, leading to higher compilation success rates and better semantic precision. As a result, the framework proves highly suitable for industrial automation applications and outperforms state-of-the-art models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01406v3">Adapting Multilingual LLMs to Low-Resource Languages with Knowledge Graphs via Adapters</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 9 pages, KaLLM workshop
    </div>
    <details class="paper-abstract">
      This paper explores the integration of graph knowledge from linguistic ontologies into multilingual Large Language Models (LLMs) using adapters to improve performance for low-resource languages (LRLs) in sentiment analysis (SA) and named entity recognition (NER). Building upon successful parameter-efficient fine-tuning techniques, such as K-ADAPTER and MAD-X, we propose a similar approach for incorporating knowledge from multilingual graphs, connecting concepts in various languages with each other through linguistic relationships, into multilingual LLMs for LRLs. Specifically, we focus on eight LRLs -- Maltese, Bulgarian, Indonesian, Nepali, Javanese, Uyghur, Tibetan, and Sinhala -- and employ language-specific adapters fine-tuned on data extracted from the language-specific section of ConceptNet, aiming to enable knowledge transfer across the languages covered by the knowledge graph. We compare various fine-tuning objectives, including standard Masked Language Modeling (MLM), MLM with full-word masking, and MLM with targeted masking, to analyse their effectiveness in learning and integrating the extracted graph data. Through empirical evaluation on language-specific tasks, we assess how structured graph knowledge affects the performance of multilingual LLMs for LRLs in SA and NER, providing insights into the potential benefits of adapting language models for low-resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13998v1">Few-shot Steerable Alignment: Adapting Rewards and LLM Policies with Neural Processes</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly embedded in everyday applications, ensuring their alignment with the diverse preferences of individual users has become a critical challenge. Currently deployed approaches typically assume homogeneous user objectives and rely on single-objective fine-tuning. However, human preferences are inherently heterogeneous, influenced by various unobservable factors, leading to conflicting signals in preference data. Existing solutions addressing this diversity often require costly datasets labelled for specific objectives and involve training multiple reward models or LLM policies, which is computationally expensive and impractical. In this work, we present a novel framework for few-shot steerable alignment, where users' underlying preferences are inferred from a small sample of their choices. To achieve this, we extend the Bradley-Terry-Luce model to handle heterogeneous preferences with unobserved variability factors and propose its practical implementation for reward modelling and LLM fine-tuning. Thanks to our proposed approach of functional parameter-space conditioning, LLMs trained with our framework can be adapted to individual preferences at inference time, generating outputs over a continuum of behavioural modes. We empirically validate the effectiveness of methods, demonstrating their ability to capture and align with diverse human preferences in a data-efficient manner. Our code is made available at: https://github.com/kasia-kobalczyk/few-shot-steerable-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09043v3">Do LLMs Play Dice? Exploring Probability Distribution Sampling in Large Language Models for Behavioral Simulation</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 The 31st International Conference on Computational Linguistics (COLING 2025)
    </div>
    <details class="paper-abstract">
      With the rapid advancement of large language models (LLMs) for handling complex language tasks, an increasing number of studies are employing LLMs as agents to emulate the sequential decision-making processes of humans often represented as Markov decision-making processes (MDPs). The actions in MDPs adhere to specific probability distributions and require iterative sampling. This arouses curiosity regarding the capacity of LLM agents to comprehend probability distributions, thereby guiding the agent's behavioral decision-making through probabilistic sampling and generating behavioral sequences. To answer the above question, we divide the problem into two main aspects: sequence simulation with known probability distribution and sequence simulation with unknown probability distribution. Our analysis indicates that LLM agents can understand probabilities, but they struggle with probability sampling. Their ability to perform probabilistic sampling can be improved to some extent by integrating coding tools, but this level of sampling precision still makes it difficult to simulate human behavior as agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13942v1">A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 25 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Disagreement in human labeling is ubiquitous, and can be captured in human judgment distributions (HJDs). Recent research has shown that explanations provide valuable information for understanding human label variation (HLV) and large language models (LLMs) can approximate HJD from a few human-provided label-explanation pairs. However, collecting explanations for every label is still time-consuming. This paper examines whether LLMs can be used to replace humans in generating explanations for approximating HJD. Specifically, we use LLMs as annotators to generate model explanations for a few given human labels. We test ways to obtain and combine these label-explanations with the goal to approximate human judgment distribution. We further compare the resulting human with model-generated explanations, and test automatic and human explanation selection. Our experiments show that LLM explanations are promising for NLI: to estimate HJD, generated explanations yield comparable results to human's when provided with human labels. Importantly, our results generalize from datasets with human explanations to i) datasets where they are not available and ii) challenging out-of-distribution test sets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13922v1">Pipeline Analysis for Developing Instruct LLMs in Low-Resource Languages: A Case Study on Basque</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are typically optimized for resource-rich languages like English, exacerbating the gap between high-resource and underrepresented languages. This work presents a detailed analysis of strategies for developing a model capable of following instructions in a low-resource language, specifically Basque, by focusing on three key stages: pre-training, instruction tuning, and alignment with human preferences. Our findings demonstrate that continual pre-training with a high-quality Basque corpus of around 600 million words improves natural language understanding (NLU) of the foundational model by over 12 points. Moreover, instruction tuning and human preference alignment using automatically translated datasets proved highly effective, resulting in a 24-point improvement in instruction-following performance. The resulting models, Llama-eus-8B and Llama-eus-8B-instruct, establish a new state-of-the-art for Basque in the sub-10B parameter category.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01638v4">TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Accepted by AAAI 2025 (Main Technical Track)
    </div>
    <details class="paper-abstract">
      Multivariate time series forecasting (MTSF) aims to learn temporal dynamics among variables to forecast future time series. Existing statistical and deep learning-based methods suffer from limited learnable parameters and small-scale training data. Recently, large language models (LLMs) combining time series with textual prompts have achieved promising performance in MTSF. However, we discovered that current LLM-based solutions fall short in learning disentangled embeddings. We introduce TimeCMA, an intuitive yet effective framework for MTSF via cross-modality alignment. Specifically, we present a dual-modality encoding with two branches: the time series encoding branch extracts disentangled yet weak time series embeddings, and the LLM-empowered encoding branch wraps the same time series with text as prompts to obtain entangled yet robust prompt embeddings. As a result, such a cross-modality alignment retrieves both disentangled and robust time series embeddings, ``the best of two worlds'', from the prompt embeddings based on time series and prompt modality similarities. As another key design, to reduce the computational costs from time series with their length textual prompts, we design an effective prompt to encourage the most essential temporal information to be encapsulated in the last token: only the last token is passed to downstream prediction. We further store the last token embeddings to accelerate inference speed. Extensive experiments on eight real datasets demonstrate that TimeCMA outperforms state-of-the-arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15380v3">Kalahi: A handcrafted, grassroots cultural LLM evaluation suite for Filipino</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Accepted for presentation at Paclic 38, 2024
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) today may not necessarily provide culturally appropriate and relevant responses to its Filipino users. We introduce Kalahi, a cultural LLM evaluation suite collaboratively created by native Filipino speakers. It is composed of 150 high-quality, handcrafted and nuanced prompts that test LLMs for generations that are relevant to shared Filipino cultural knowledge and values. Strong LLM performance in Kalahi indicates a model's ability to generate responses similar to what an average Filipino would say or do in a given situation. We conducted experiments on LLMs with multilingual and Filipino language support. Results show that Kalahi, while trivial for Filipinos, is challenging for LLMs, with the best model answering only 46.0% of the questions correctly compared to native Filipino performance of 89.10%. Thus, Kalahi can be used to accurately and reliably evaluate Filipino cultural representation in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18416v3">PersonaGym: Evaluating Persona Agents and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 21 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Persona agents, which are LLM agents that act according to an assigned persona, have demonstrated impressive contextual response capabilities across various applications. These persona agents offer significant enhancements across diverse sectors, such as education, healthcare, and entertainment, where model developers can align agent responses to different user requirements thereby broadening the scope of agent applications. However, evaluating persona agent performance is incredibly challenging due to the complexity of assessing persona adherence in free-form interactions across various environments that are relevant to each persona agent. We introduce PersonaGym, the first dynamic evaluation framework for assessing persona agents, and PersonaScore, the first automated human-aligned metric grounded in decision theory for comprehensive large-scale evaluation of persona agents. Our evaluation of 6 open and closed-source LLMs, using a benchmark encompassing 200 personas and 10,000 questions, reveals significant opportunities for advancement in persona agent capabilities across state-of-the-art models. For example, Claude 3.5 Sonnet only has a 2.97% relative improvement in PersonaScore than GPT 3.5 despite being a much more advanced model. Importantly, we find that increased model size and complexity do not necessarily imply enhanced persona agent capabilities thereby highlighting the pressing need for algorithmic and architectural invention towards faithful and performant persona agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13178v2">SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 21 pages, 14 tables, 7 figures, submitted to ICRA 2024
    </div>
    <details class="paper-abstract">
      With the integration of large language models (LLMs), embodied agents have strong capabilities to execute complicated instructions in natural language, paving a way for the potential deployment of embodied robots. However, a foreseeable issue is that those embodied agents can also flawlessly execute some hazardous tasks, potentially causing damages in real world. To study this issue, we present SafeAgentBench -- a new benchmark for safety-aware task planning of embodied LLM agents. SafeAgentBench includes: (1) a new dataset with 750 tasks, covering 10 potential hazards and 3 task types; (2) SafeAgentEnv, a universal embodied environment with a low-level controller, supporting multi-agent execution with 17 high-level actions for 8 state-of-the-art baselines; and (3) reliable evaluation methods from both execution and semantic perspectives. Experimental results show that the best-performing baseline gets 69% success rate for safe tasks, but only 5% rejection rate for hazardous tasks, indicating significant safety risks. More details and codes are available at https://github.com/shengyin1224/SafeAgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11156v4">DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Sequential recommendation (SR) tasks aim to predict users' next interaction by learning their behavior sequence and capturing the connection between users' past interactions and their changing preferences. Conventional SR models often focus solely on capturing sequential patterns within the training data, neglecting the broader context and semantic information embedded in item titles from external sources. This limits their predictive power and adaptability. Large language models (LLMs) have recently shown promise in SR tasks due to their advanced understanding capabilities and strong generalization abilities. Researchers have attempted to enhance LLMs-based recommendation performance by incorporating information from conventional SR models. However, previous approaches have encountered problems such as 1) limited textual information leading to poor recommendation performance, 2) incomplete understanding and utilization of conventional SR model information by LLMs, and 3) excessive complexity and low interpretability of LLMs-based methods. To improve the performance of LLMs-based SR, we propose a novel framework, Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation (DELRec), which aims to extract knowledge from conventional SR models and enable LLMs to easily comprehend and utilize the extracted knowledge for more effective SRs. DELRec consists of two main stages: 1) Distill Pattern from Conventional SR Models, focusing on extracting behavioral patterns exhibited by conventional SR models using soft prompts through two well-designed strategies; 2) LLMs-based Sequential Recommendation, aiming to fine-tune LLMs to effectively use the distilled auxiliary information to perform SR tasks. Extensive experimental results conducted on four real datasets validate the effectiveness of the DELRec framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13774v1">Designing an LLM-Based Copilot for Manufacturing Equipment Selection</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Preprint submitted to Manufacturing Letters (MFGLET)
    </div>
    <details class="paper-abstract">
      Effective decision-making in automation equipment selection is critical for reducing ramp-up time and maintaining production quality, especially in the face of increasing product variation and market demands. However, limited expertise and resource constraints often result in inefficiencies during the ramp-up phase when new products are integrated into production lines. Existing methods often lack structured and tailored solutions to support automation engineers in reducing ramp-up time, leading to compromises in quality. This research investigates whether large-language models (LLMs), combined with Retrieval-Augmented Generation (RAG), can assist in streamlining equipment selection in ramp-up planning. We propose a factual-driven copilot integrating LLMs with structured and semi-structured knowledge retrieval for three component types (robots, feeders and vision systems), providing a guided and traceable state-machine process for decision-making in automation equipment selection. The system was demonstrated to an industrial partner, who tested it on three internal use-cases. Their feedback affirmed its capability to provide logical and actionable recommendations for automation equipment. More specifically, among 22 equipment prompts analyzed, 19 involved selecting the correct equipment while considering most requirements, and in 6 cases, all requirements were fully met.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04503v2">When LLMs Play the Telephone Game: Cumulative Changes and Attractors in Iterated Cultural Transmissions</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Code available at https://github.com/jeremyperez2/TelephoneGameLLM. Companion website with a Data Explorer tool at https://sites.google.com/view/telephone-game-llm
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) start interacting with each other and generating an increasing amount of text online, it becomes crucial to better understand how information is transformed as it passes from one LLM to the next. While significant research has examined individual LLM behaviors, existing studies have largely overlooked the collective behaviors and information distortions arising from iterated LLM interactions. Small biases, negligible at the single output level, risk being amplified in iterated interactions, potentially leading the content to evolve towards attractor states. In a series of telephone game experiments, we apply a transmission chain design borrowed from the human cultural evolution literature: LLM agents iteratively receive, produce, and transmit texts from the previous to the next agent in the chain. By tracking the evolution of text toxicity, positivity, difficulty, and length across transmission chains, we uncover the existence of biases and attractors, and study their dependence on the initial text, the instructions, language model, and model size. For instance, we find that more open-ended instructions lead to stronger attraction effects compared to more constrained tasks. We also find that different text properties display different sensitivity to attraction effects, with toxicity leading to stronger attractors than length. These findings highlight the importance of accounting for multi-step transmission dynamics and represent a first step towards a more comprehensive understanding of LLM cultural dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13705v1">Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14215v1">Generative AI Toolkit -- a framework for increasing the quality of LLM-based applications over their whole life cycle</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 16 pages, 6 figures. For source code see https://github.com/awslabs/generative-ai-toolkit
    </div>
    <details class="paper-abstract">
      As LLM-based applications reach millions of customers, ensuring their scalability and continuous quality improvement is critical for success. However, the current workflows for developing, maintaining, and operating (DevOps) these applications are predominantly manual, slow, and based on trial-and-error. With this paper we introduce the Generative AI Toolkit, which automates essential workflows over the whole life cycle of LLM-based applications. The toolkit helps to configure, test, continuously monitor and optimize Generative AI applications such as agents, thus significantly improving quality while shortening release cycles. We showcase the effectiveness of our toolkit on representative use cases, share best practices, and outline future enhancements. Since we are convinced that our Generative AI Toolkit is helpful for other teams, we are open sourcing it on and hope that others will use, forward, adapt and improve
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13667v1">Exploring Multi-Modal Integration with Tool-Augmented LLM Agents for Precise Causal Discovery</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Causal inference is an imperative foundation for decision-making across domains, such as smart health, AI for drug discovery and AIOps. Traditional statistical causal discovery methods, while well-established, predominantly rely on observational data and often overlook the semantic cues inherent in cause-and-effect relationships. The advent of Large Language Models (LLMs) has ushered in an affordable way of leveraging the semantic cues for knowledge-driven causal discovery, but the development of LLMs for causal discovery lags behind other areas, particularly in the exploration of multi-modality data. To bridge the gap, we introduce MATMCD, a multi-agent system powered by tool-augmented LLMs. MATMCD has two key agents: a Data Augmentation agent that retrieves and processes modality-augmented data, and a Causal Constraint agent that integrates multi-modal data for knowledge-driven inference. Delicate design of the inner-workings ensures successful cooperation of the agents. Our empirical study across seven datasets suggests the significant potential of multi-modality enhanced causal discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13666v1">Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      The capabilities of recent large language models (LLMs) to generate high-quality content indistinguishable by humans from human-written texts rises many concerns regarding their misuse. Previous research has shown that LLMs can be effectively misused for generating disinformation news articles following predefined narratives. Their capabilities to generate personalized (in various aspects) content have also been evaluated and mostly found usable. However, a combination of personalization and disinformation abilities of LLMs has not been comprehensively studied yet. Such a dangerous combination should trigger integrated safety filters of the LLMs, if there are some. This study fills this gap by evaluation of vulnerabilities of recent open and closed LLMs, and their willingness to generate personalized disinformation news articles in English. We further explore whether the LLMs can reliably meta-evaluate the personalization quality and whether the personalization affects the generated-texts detectability. Our results demonstrate the need for stronger safety-filters and disclaimers, as those are not properly functioning in most of the evaluated LLMs. Additionally, our study revealed that the personalization actually reduces the safety-filter activations; thus effectively functioning as a jailbreak. Such behavior must be urgently addressed by LLM developers and service providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13660v1">PsyDT: Using LLMs to Construct the Digital Twin of Psychological Counselor with Personalized Counseling Style for Psychological Counseling</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Currently, large language models (LLMs) have made significant progress in the field of psychological counseling. However, existing mental health LLMs overlook a critical issue where they do not consider the fact that different psychological counselors exhibit different personal styles, including linguistic style and therapy techniques, etc. As a result, these LLMs fail to satisfy the individual needs of clients who seek different counseling styles. To help bridge this gap, we propose PsyDT, a novel framework using LLMs to construct the Digital Twin of Psychological counselor with personalized counseling style. Compared to the time-consuming and costly approach of collecting a large number of real-world counseling cases to create a specific counselor's digital twin, our framework offers a faster and more cost-effective solution. To construct PsyDT, we utilize dynamic one-shot learning by using GPT-4 to capture counselor's unique counseling style, mainly focusing on linguistic style and therapy techniques. Subsequently, using existing single-turn long-text dialogues with client's questions, GPT-4 is guided to synthesize multi-turn dialogues of specific counselor. Finally, we fine-tune the LLMs on the synthetic dataset, PsyDTCorpus, to achieve the digital twin of psychological counselor with personalized counseling style. Experimental results indicate that our proposed PsyDT framework can synthesize multi-turn dialogues that closely resemble real-world counseling cases and demonstrate better performance compared to other baselines, thereby show that our framework can effectively construct the digital twin of psychological counselor with a specific counseling style.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13602v1">Beyond Outcomes: Transparent Assessment of LLM Reasoning in Games</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in real-world applications that demand complex reasoning. To track progress, robust benchmarks are required to evaluate their capabilities beyond superficial pattern recognition. However, current LLM reasoning benchmarks often face challenges such as insufficient interpretability, performance saturation or data contamination. To address these challenges, we introduce GAMEBoT, a gaming arena designed for rigorous and transparent assessment of LLM reasoning capabilities. GAMEBoT decomposes complex reasoning in games into predefined modular subproblems. This decomposition allows us to design a suite of Chain-of-Thought (CoT) prompts that leverage domain knowledge to guide LLMs in addressing these subproblems before action selection. Furthermore, we develop a suite of rule-based algorithms to generate ground truth for these subproblems, enabling rigorous validation of the LLMs' intermediate reasoning steps. This approach facilitates evaluation of both the quality of final actions and the accuracy of the underlying reasoning process. GAMEBoT also naturally alleviates the risk of data contamination through dynamic games and head-to-head LLM competitions. We benchmark 17 prominent LLMs across eight games, encompassing various strategic abilities and game characteristics. Our results suggest that GAMEBoT presents a significant challenge, even when LLMs are provided with detailed CoT prompts. Project page: \url{https://visual-ai.github.io/gamebot}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13582v1">EvoWiki: Evaluating LLMs on Evolving Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Knowledge utilization is a critical aspect of LLMs, and understanding how they adapt to evolving knowledge is essential for their effective deployment. However, existing benchmarks are predominantly static, failing to capture the evolving nature of LLMs and knowledge, leading to inaccuracies and vulnerabilities such as contamination. In this paper, we introduce EvoWiki, an evolving dataset designed to reflect knowledge evolution by categorizing information into stable, evolved, and uncharted states. EvoWiki is fully auto-updatable, enabling precise evaluation of continuously changing knowledge and newly released LLMs. Through experiments with Retrieval-Augmented Generation (RAG) and Contunual Learning (CL), we evaluate how effectively LLMs adapt to evolving knowledge. Our results indicate that current models often struggle with evolved knowledge, frequently providing outdated or incorrect responses. Moreover, the dataset highlights a synergistic effect between RAG and CL, demonstrating their potential to better adapt to evolving knowledge. EvoWiki provides a robust benchmark for advancing future research on the knowledge evolution capabilities of large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13578v1">Socio-Culturally Aware Evaluation Framework for LLM-Based Content Moderation</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 Accepted in SUMEval Workshop in COLING 2025
    </div>
    <details class="paper-abstract">
      With the growth of social media and large language models, content moderation has become crucial. Many existing datasets lack adequate representation of different groups, resulting in unreliable assessments. To tackle this, we propose a socio-culturally aware evaluation framework for LLM-driven content moderation and introduce a scalable method for creating diverse datasets using persona-based generation. Our analysis reveals that these datasets provide broader perspectives and pose greater challenges for LLMs than diversity-focused generation methods without personas. This challenge is especially pronounced in smaller LLMs, emphasizing the difficulties they encounter in moderating such diverse content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16216v1">GraphLoRA: Empowering LLMs Fine-Tuning via Graph Collaboration of MoE</a></div>
    <div class="paper-meta">
      📅 2024-12-18
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method that has been widely adopted in various downstream applications of LLMs. Together with the Mixture-of-Expert (MoE) technique, fine-tuning approaches have shown remarkable improvements in model capability. However, the coordination of multiple experts in existing studies solely relies on the weights assigned by the simple router function. Lack of communication and collaboration among experts exacerbate the instability of LLMs due to the imbalance load problem of MoE. To address this issue, we propose a novel MoE graph-based LLM fine-tuning framework GraphLoRA, in which a graph router function is designed to capture the collaboration signals among experts by graph neural networks (GNNs). GraphLoRA enables all experts to understand input knowledge and share information from neighbor experts by aggregating operations. Besides, to enhance each expert's capability and their collaborations, we design two novel coordination strategies: the Poisson distribution-based distinction strategy and the Normal distribution-based load balance strategy. Extensive experiments on four real-world datasets demonstrate the effectiveness of our GraphLoRA in parameter-efficient fine-tuning of LLMs, showing the benefits of facilitating collaborations of multiple experts in the graph router of GraphLoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02220v3">Data to Defense: The Role of Curation in Customizing LLMs Against Jailbreaking Attacks</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely adapted for downstream applications through fine-tuning, a process named customization. However, recent studies have identified a vulnerability during this process, where malicious samples can compromise the robustness of LLMs and amplify harmful behaviors-an attack commonly referred to as jailbreaking. To address this challenge, we propose an adaptive data curation approach allowing any text to be curated to enhance its effectiveness in counteracting harmful samples during customization. To avoid the need for additional defensive modules, we further introduce a comprehensive mitigation framework spanning the lifecycle of the customization process: before customization to immunize LLMs against future jailbreak attempts, during customization to neutralize risks, and after customization to restore compromised models. Experimental results demonstrate a significant reduction in jailbreaking effects, achieving up to a 100% success rate in generating safe responses. By combining adaptive data curation with lifecycle-based mitigation strategies, this work represents a solid step forward in mitigating jailbreaking risks and ensuring the secure adaptation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16215v1">Zero-Shot Image Moderation in Google Ads with LLM-Assisted Textual Descriptions and Cross-modal Co-embeddings</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      We present a scalable and agile approach for ads image content moderation at Google, addressing the challenges of moderating massive volumes of ads with diverse content and evolving policies. The proposed method utilizes human-curated textual descriptions and cross-modal text-image co-embeddings to enable zero-shot classification of policy violating ads images, bypassing the need for extensive supervised training data and human labeling. By leveraging large language models (LLMs) and user expertise, the system generates and refines a comprehensive set of textual descriptions representing policy guidelines. During inference, co-embedding similarity between incoming images and the textual descriptions serves as a reliable signal for policy violation detection, enabling efficient and adaptable ads content moderation. Evaluation results demonstrate the efficacy of this framework in significantly boosting the detection of policy violating content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13420v1">BotSim: LLM-Powered Malicious Social Botnet Simulation</a></div>
    <div class="paper-meta">
      📅 2024-12-18
    </div>
    <details class="paper-abstract">
      Social media platforms like X(Twitter) and Reddit are vital to global communication. However, advancements in Large Language Model (LLM) technology give rise to social media bots with unprecedented intelligence. These bots adeptly simulate human profiles, conversations, and interactions, disseminating large amounts of false information and posing significant challenges to platform regulation. To better understand and counter these threats, we innovatively design BotSim, a malicious social botnet simulation powered by LLM. BotSim mimics the information dissemination patterns of real-world social networks, creating a virtual environment composed of intelligent agent bots and real human users. In the temporal simulation constructed by BotSim, these advanced agent bots autonomously engage in social interactions such as posting and commenting, effectively modeling scenarios of information flow and user interaction. Building on the BotSim framework, we construct a highly human-like, LLM-driven bot dataset called BotSim-24 and benchmark multiple bot detection strategies against it. The experimental results indicate that detection methods effective on traditional bot datasets perform worse on BotSim-24, highlighting the urgent need for new detection strategies to address the cybersecurity threats posed by these advanced bots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12509v1">Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become increasingly powerful and ubiquitous, but their stochastic nature poses challenges to the reliability of their outputs. While deterministic settings can improve consistency, they do not guarantee reliability, as a single sample from the model's probability distribution can still be misleading. Building upon the concept of LLM-as-a-judge, we introduce a novel framework for rigorously evaluating the reliability of LLM judgments, leveraging McDonald's omega. We evaluate the reliability of LLMs when judging the outputs of other LLMs on standard single-turn and multi-turn benchmarks, simultaneously investigating the impact of temperature on reliability. By analyzing these results, we demonstrate the limitations of fixed randomness and the importance of considering multiple samples, which we show has significant implications for downstream applications. Our findings highlight the need for a nuanced understanding of LLM reliability and the potential risks associated with over-reliance on single-shot evaluations. This work provides a crucial step towards building more trustworthy and reliable LLM-based systems and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12504v1">Boosting LLM-based Relevance Modeling with Distribution-Aware Robust Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      With the rapid advancement of pre-trained large language models (LLMs), recent endeavors have leveraged the capabilities of LLMs in relevance modeling, resulting in enhanced performance. This is usually done through the process of fine-tuning LLMs on specifically annotated datasets to determine the relevance between queries and items. However, there are two limitations when LLMs are naively employed for relevance modeling through fine-tuning and inference. First, it is not inherently efficient for performing nuanced tasks beyond simple yes or no answers, such as assessing search relevance. It may therefore tend to be overconfident and struggle to distinguish fine-grained degrees of relevance (e.g., strong relevance, weak relevance, irrelevance) used in search engines. Second, it exhibits significant performance degradation when confronted with data distribution shift in real-world scenarios. In this paper, we propose a novel Distribution-Aware Robust Learning framework (DaRL) for relevance modeling in Alipay Search. Specifically, we design an effective loss function to enhance the discriminability of LLM-based relevance modeling across various fine-grained degrees of query-item relevance. To improve the generalizability of LLM-based relevance modeling, we first propose the Distribution-Aware Sample Augmentation (DASA) module. This module utilizes out-of-distribution (OOD) detection techniques to actively select appropriate samples that are not well covered by the original training set for model fine-tuning. Furthermore, we adopt a multi-stage fine-tuning strategy to simultaneously improve in-distribution (ID) and OOD performance, bridging the performance gap between them. DaRL has been deployed online to serve the Alipay's insurance product search...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12488v1">A System for Microserving of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      The recent advances in LLMs bring a strong demand for efficient system support to improve overall serving efficiency. As LLM inference scales towards multiple GPUs and even multiple compute nodes, various coordination patterns, such as prefill-decode disaggregation and context migration, arise in serving systems. Most inference services today expose a coarse-grained request-level API with a pre-configured coordination strategy, limiting the ability to customize and dynamically reconfigure the coordination. In this paper, we propose LLM microserving, a multi-level architecture for structuring and programming LLM inference services. We introduces simple yet effective microserving APIs to support fine-grained sub-request level actions. A programmable router transforms user requests into sub-request calls, enabling the dynamic reconfiguration of serving patterns. To support diverse execution patterns, we develop a unified KV cache interface that handles various KV compute, transfer, and reuse scenarios. Our evaluation shows that LLM microserving can be reconfigured to support multiple disaggregation orchestration strategies in a few lines of Python code while maintaining state-of-the-art performance for LLM inference tasks. Additionally, it allows us to explore new strategy variants that reduce up to 47% of job completion time compared to the existing strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12464v1">LLM is Knowledge Graph Reasoner: LLM's Intuition-aware Knowledge Graph Reasoning for Cold-start Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Accepted to the 47th European Conference on Information Retrieval (ECIR2025)
    </div>
    <details class="paper-abstract">
      Knowledge Graphs (KGs) represent relationships between entities in a graph structure and have been widely studied as promising tools for realizing recommendations that consider the accurate content information of items. However, traditional KG-based recommendation methods face fundamental challenges: insufficient consideration of temporal information and poor performance in cold-start scenarios. On the other hand, Large Language Models (LLMs) can be considered databases with a wealth of knowledge learned from the web data, and they have recently gained attention due to their potential application as recommendation systems. Although approaches that treat LLMs as recommendation systems can leverage LLMs' high recommendation literacy, their input token limitations make it impractical to consider the entire recommendation domain dataset and result in scalability issues. To address these challenges, we propose a LLM's Intuition-aware Knowledge graph Reasoning model (LIKR). Our main idea is to treat LLMs as reasoners that output intuitive exploration strategies for KGs. To integrate the knowledge of LLMs and KGs, we trained a recommendation agent through reinforcement learning using a reward function that integrates different recommendation strategies, including LLM's intuition and KG embeddings. By incorporating temporal awareness through prompt engineering and generating textual representations of user preferences from limited interactions, LIKR can improve recommendation performance in cold-start scenarios. Furthermore, LIKR can avoid scalability issues by using KGs to represent recommendation domain datasets and limiting the LLM's output to KG exploration strategies. Experiments on real-world datasets demonstrate that our model outperforms state-of-the-art recommendation methods in cold-start sequential recommendation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12459v1">LITA: An Efficient LLM-assisted Iterative Topic Augmentation Framework</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Topic modeling is widely used for uncovering thematic structures within text corpora, yet traditional models often struggle with specificity and coherence in domain-focused applications. Guided approaches, such as SeededLDA and CorEx, incorporate user-provided seed words to improve relevance but remain labor-intensive and static. Large language models (LLMs) offer potential for dynamic topic refinement and discovery, yet their application often incurs high API costs. To address these challenges, we propose the LLM-assisted Iterative Topic Augmentation framework (LITA), an LLM-assisted approach that integrates user-provided seeds with embedding-based clustering and iterative refinement. LITA identifies a small number of ambiguous documents and employs an LLM to reassign them to existing or new topics, minimizing API costs while enhancing topic quality. Experiments on two datasets across topic quality and clustering performance metrics demonstrate that LITA outperforms five baseline models, including LDA, SeededLDA, CorEx, BERTopic, and PromptTopic. Our work offers an efficient and adaptable framework for advancing topic modeling and text clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12456v1">Graph Learning in the Era of LLMs: A Survey from the Perspective of Data, Models, and Tasks</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 In progress
    </div>
    <details class="paper-abstract">
      With the increasing prevalence of cross-domain Text-Attributed Graph (TAG) Data (e.g., citation networks, recommendation systems, social networks, and ai4science), the integration of Graph Neural Networks (GNNs) and Large Language Models (LLMs) into a unified Model architecture (e.g., LLM as enhancer, LLM as collaborators, LLM as predictor) has emerged as a promising technological paradigm. The core of this new graph learning paradigm lies in the synergistic combination of GNNs' ability to capture complex structural relationships and LLMs' proficiency in understanding informative contexts from the rich textual descriptions of graphs. Therefore, we can leverage graph description texts with rich semantic context to fundamentally enhance Data quality, thereby improving the representational capacity of model-centric approaches in line with data-centric machine learning principles. By leveraging the strengths of these distinct neural network architectures, this integrated approach addresses a wide range of TAG-based Task (e.g., graph learning, graph reasoning, and graph question answering), particularly in complex industrial scenarios (e.g., supervised, few-shot, and zero-shot settings). In other words, we can treat text as a medium to enable cross-domain generalization of graph learning Model, allowing a single graph model to effectively handle the diversity of downstream graph-based Task across different data domains. This work serves as a foundational reference for researchers and practitioners looking to advance graph learning methodologies in the rapidly evolving landscape of LLM. We consistently maintain the related open-source materials at \url{https://github.com/xkLi-Allen/Awesome-GNN-in-LLMs-Papers}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13381v1">An Automated Explainable Educational Assessment System Built on LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Accepted to AAAI 2025
    </div>
    <details class="paper-abstract">
      In this demo, we present AERA Chat, an automated and explainable educational assessment system designed for interactive and visual evaluations of student responses. This system leverages large language models (LLMs) to generate automated marking and rationale explanations, addressing the challenge of limited explainability in automated educational assessment and the high costs associated with annotation. Our system allows users to input questions and student answers, providing educators and researchers with insights into assessment accuracy and the quality of LLM-assessed rationales. Additionally, it offers advanced visualization and robust evaluation tools, enhancing the usability for educational assessment and facilitating efficient rationale verification. Our demo video can be found at https://youtu.be/qUSjz-sxlBc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13337v1">Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 33 pages, 19 figures. Appendix included in submission. Submitted to ICLR 2025
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has created a significant disparity: industrial research labs with their computational resources, expert teams, and advanced infrastructures, can effectively fine-tune LLMs, while individual developers and small organizations face barriers due to limited resources. In this paper, we aim to bridge this gap by presenting a comprehensive study on supervised fine-tuning of LLMs using instruction-tuning datasets spanning diverse knowledge domains and skills. We focus on small-sized LLMs (3B to 7B parameters) for their cost-efficiency and accessibility. We explore various training configurations and strategies across four open-source pre-trained models. We provide detailed documentation of these configurations, revealing findings that challenge several common training practices, including hyperparameter recommendations from TULU and phased training recommended by Orca. Key insights from our work include: (i) larger batch sizes paired with lower learning rates lead to improved model performance on benchmarks such as MMLU, MTBench, and Open LLM Leaderboard; (ii) early-stage training dynamics, such as lower gradient norms and higher loss values, are strong indicators of better final model performance, enabling early termination of sub-optimal runs and significant computational savings; (iii) through a thorough exploration of hyperparameters like warmup steps and learning rate schedules, we provide guidance for practitioners and find that certain simplifications do not compromise performance; and (iv) we observed no significant difference in performance between phased and stacked training strategies, but stacked training is simpler and more sample efficient. With these findings holding robustly across datasets and models, we hope this study serves as a guide for practitioners fine-tuning small LLMs and promotes a more inclusive environment for LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11404v3">AraDiCE: Benchmarks for Dialectal and Cultural Capabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Benchmarking, Culturally Informed, Large Language Models, Arabic NLP, LLMs, Arabic Dialect, Dialectal Benchmarking
    </div>
    <details class="paper-abstract">
      Arabic, with its rich diversity of dialects, remains significantly underrepresented in Large Language Models, particularly in dialectal variations. We address this gap by introducing seven synthetic datasets in dialects alongside Modern Standard Arabic (MSA), created using Machine Translation (MT) combined with human post-editing. We present AraDiCE, a benchmark for Arabic Dialect and Cultural Evaluation. We evaluate LLMs on dialect comprehension and generation, focusing specifically on low-resource Arabic dialects. Additionally, we introduce the first-ever fine-grained benchmark designed to evaluate cultural awareness across the Gulf, Egypt, and Levant regions, providing a novel dimension to LLM evaluation. Our findings demonstrate that while Arabic-specific models like Jais and AceGPT outperform multilingual models on dialectal tasks, significant challenges persist in dialect identification, generation, and translation. This work contributes $\approx$45K post-edited samples, a cultural benchmark, and highlights the importance of tailored training to improve LLM performance in capturing the nuances of diverse Arabic dialects and cultural contexts. We have released the dialectal translation models and benchmarks developed in this study (https://huggingface.co/datasets/QCRI/AraDiCE).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04755v3">LLM Whisperer: An Inconspicuous Attack to Bias LLM Responses</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Writing effective prompts for large language models (LLM) can be unintuitive and burdensome. In response, services that optimize or suggest prompts have emerged. While such services can reduce user effort, they also introduce a risk: the prompt provider can subtly manipulate prompts to produce heavily biased LLM responses. In this work, we show that subtle synonym replacements in prompts can increase the likelihood (by a difference up to 78%) that LLMs mention a target concept (e.g., a brand, political party, nation). We substantiate our observations through a user study, showing our adversarially perturbed prompts 1) are indistinguishable from unaltered prompts by humans, 2) push LLMs to recommend target concepts more often, and 3) make users more likely to notice target concepts, all without arousing suspicion. The practicality of this attack has the potential to undermine user autonomy. Among other measures, we recommend implementing warnings against using prompts from untrusted parties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15275v1">Fooling LLM graders into giving better grades through neural activity guided adversarial prompting</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 16 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The deployment of artificial intelligence (AI) in critical decision-making and evaluation processes raises concerns about inherent biases that malicious actors could exploit to distort decision outcomes. We propose a systematic method to reveal such biases in AI evaluation systems and apply it to automated essay grading as an example. Our approach first identifies hidden neural activity patterns that predict distorted decision outcomes and then optimizes an adversarial input suffix to amplify such patterns. We demonstrate that this combination can effectively fool large language model (LLM) graders into assigning much higher grades than humans would. We further show that this white-box attack transfers to black-box attacks on other models, including commercial closed-source models like Gemini. They further reveal the existence of a "magic word" that plays a pivotal role in the efficacy of the attack. We trace the origin of this magic word bias to the structure of commonly-used chat templates for supervised fine-tuning of LLMs and show that a minor change in the template can drastically reduce the bias. This work not only uncovers vulnerabilities in current LLMs but also proposes a systematic method to identify and remove hidden biases, contributing to the goal of ensuring AI safety and security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10370v2">Let's Get to the Point: LLM-Supported Planning, Drafting, and Revising of Research-Paper Blog Posts</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Research-paper blog posts help scientists to disseminate their work to a larger audience, but translating scientific long documents into long-form summaries like blog posts raises unique challenges: 1) planning what paper content to include in the blog post, 2) drafting the selected content in sections amenable to a paper blog post, and 3) revising the blog post to be scientifically accurate but also concise, easy to understand, and engaging. Can we harness the power of large language models (LLMs) to assist researchers with these challenges? To investigate this question, we developed Papers-to-Posts, an LLM-powered tool that implements a new Plan-Draft-Revise workflow for mixed-initiative long-form paper summarization. An LLM-generated paper outline with pre-selected yet adjustable bullet points helps users to plan what information to include. Meanwhile, customizable LLM instructions support drafting the text with a suitable structure and revising the text to have an appropriate tone. Through two studies, we compared Papers-to-Posts to a strong baseline tool that provides an LLM-generated draft and access to free-form LLM prompting, and we found that Papers-to-Posts improved researchers' editing power. In a within-subjects lab study (N=20 participants), Papers-to-Posts led participants to make significantly more change to initial LLM drafts within a fixed amount of time and to be significantly more satisfied with their final blog post, without increasing cognitive load. Furthermore, in a between-subjects deployment study (N=37 blog posts, 26 participants), Papers-to-Posts led participants to make more change to initial LLM drafts within a given amount of time as well as writing actions, without decreasing satisfaction with the final blog posts or increasing cognitive load.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10400v2">Reinforcement Learning Enhanced LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      This paper surveys research in the rapidly growing field of enhancing large language models (LLMs) with reinforcement learning (RL), a technique that enables LLMs to improve their performance by receiving feedback in the form of rewards based on the quality of their outputs, allowing them to generate more accurate, coherent, and contextually appropriate responses. In this work, we make a systematic review of the most up-to-date state of knowledge on RL-enhanced LLMs, attempting to consolidate and analyze the rapidly growing research in this field, helping researchers understand the current challenges and advancements. Specifically, we (1) detail the basics of RL; (2) introduce popular RL-enhanced LLMs; (3) review researches on two widely-used reward model-based RL techniques: Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning from AI Feedback (RLAIF); and (4) explore Direct Preference Optimization (DPO), a set of methods that bypass the reward model to directly use human preference data for aligning LLM outputs with human expectations. We will also point out current challenges and deficiencies of existing methods and suggest some avenues for further improvements. Project page of this work can be found at: \url{https://github.com/ShuheWang1998/Reinforcement-Learning-Enhanced-LLMs-A-Survey}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04049v3">Systematic Biases in LLM Simulations of Debates</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Published as a conference paper at EMNLP 2024
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs), has opened exciting possibilities for constructing computational simulations designed to replicate human behavior accurately. Current research suggests that LLM-based agents become increasingly human-like in their performance, sparking interest in using these AI agents as substitutes for human participants in behavioral studies. However, LLMs are complex statistical learners without straightforward deductive rules, making them prone to unexpected behaviors. Hence, it is crucial to study and pinpoint the key behavioral distinctions between humans and LLM-based agents. In this study, we highlight the limitations of LLMs in simulating human interactions, particularly focusing on LLMs' ability to simulate political debates on topics that are important aspects of people's day-to-day lives and decision-making processes. Our findings indicate a tendency for LLM agents to conform to the model's inherent social biases despite being directed to debate from certain political perspectives. This tendency results in behavioral patterns that seem to deviate from well-established social dynamics among humans. We reinforce these observations using an automatic self-fine-tuning method, which enables us to manipulate the biases within the LLM and demonstrate that agents subsequently align with the altered biases. These results underscore the need for further research to develop methods that help agents overcome these biases, a critical step toward creating more realistic simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13103v1">AI PERSONA: Towards Life-long Personalization of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      In this work, we introduce the task of life-long personalization of large language models. While recent mainstream efforts in the LLM community mainly focus on scaling data and compute for improved capabilities of LLMs, we argue that it is also very important to enable LLM systems, or language agents, to continuously adapt to the diverse and ever-changing profiles of every distinct user and provide up-to-date personalized assistance. We provide a clear task formulation and introduce a simple, general, effective, and scalable framework for life-long personalization of LLM systems and language agents. To facilitate future research on LLM personalization, we also introduce methods to synthesize realistic benchmarks and robust evaluation metrics. We will release all codes and data for building and benchmarking life-long personalized LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09739v2">PersonaMark: Personalized LLM watermarking for model protection and user attribution</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rapid advancement of customized Large Language Models (LLMs) offers considerable convenience. However, it also intensifies concerns regarding the protection of copyright/confidential information. With the extensive adoption of private LLMs, safeguarding model copyright and ensuring data privacy have become critical. Text watermarking has emerged as a viable solution for detecting AI-generated content and protecting models. However, existing methods fall short in providing individualized watermarks for each user, a critical feature for enhancing accountability and traceability. In this paper, we introduce PersonaMark, a novel personalized text watermarking scheme designed to protect LLMs' copyrights and bolster accountability. PersonaMark leverages sentence structure as a subtle carrier of watermark information and optimizes the generation process to maintain the natural output of the model. By employing a personalized hashing function, unique watermarks are embedded for each user, enabling high-quality text generation without compromising the model's performance. This approach is both time-efficient and scalable, capable of handling large numbers of users through a multi-user hashing mechanism. To the best of our knowledge, this is a pioneer study to explore personalized watermarking in LLMs. We conduct extensive evaluations across four LLMs, analyzing various metrics such as perplexity, sentiment, alignment, and readability. The results validate that PersonaMark preserves text quality, ensures unbiased watermark insertion, and offers robust watermark detection capabilities, all while maintaining the model's behavior with minimal disruption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12957v2">Towards Reliable Latent Knowledge Estimation in LLMs: Zero-Prompt Many-Shot Based Factual Knowledge Extraction</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      In this paper, we focus on the challenging task of reliably estimating factual knowledge that is embedded inside large language models (LLMs). To avoid reliability concerns with prior approaches, we propose to eliminate prompt engineering when probing LLMs for factual knowledge. Our approach, called Zero-Prompt Latent Knowledge Estimator (ZP-LKE), leverages the in-context learning ability of LLMs to communicate both the factual knowledge question as well as the expected answer format. Our knowledge estimator is both conceptually simpler (i.e., doesn't depend on meta-linguistic judgments of LLMs) and easier to apply (i.e., is not LLM-specific), and we demonstrate that it can surface more of the latent knowledge embedded in LLMs. We also investigate how different design choices affect the performance of ZP-LKE. Using the proposed estimator, we perform a large-scale evaluation of the factual knowledge of a variety of open-source LLMs, like OPT, Pythia, Llama(2), Mistral, Gemma, etc. over a large set of relations and facts from the Wikidata knowledge base. We observe differences in the factual knowledge between different model families and models of different sizes, that some relations are consistently better known than others but that models differ in the precise facts they know, and differences in the knowledge of base models and their finetuned counterparts. Code available at: https://github.com/QinyuanWu0710/ZeroPrompt_LKE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12981v1">Unlocking LLMs: Addressing Scarce Data and Bias Challenges in Mental Health</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 International Conference on Natural Language Processing and Artificial Intelligence for Cyber Security (NLPAICS) 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising capabilities in healthcare analysis but face several challenges like hallucinations, parroting, and bias manifestation. These challenges are exacerbated in complex, sensitive, and low-resource domains. Therefore, in this work we introduce IC-AnnoMI, an expert-annotated motivational interviewing (MI) dataset built upon AnnoMI by generating in-context conversational dialogues leveraging LLMs, particularly ChatGPT. IC-AnnoMI employs targeted prompts accurately engineered through cues and tailored information, taking into account therapy style (empathy, reflection), contextual relevance, and false semantic change. Subsequently, the dialogues are annotated by experts, strictly adhering to the Motivational Interviewing Skills Code (MISC), focusing on both the psychological and linguistic dimensions of MI dialogues. We comprehensively evaluate the IC-AnnoMI dataset and ChatGPT's emotional reasoning ability and understanding of domain intricacies by modeling novel classification tasks employing several classical machine learning and current state-of-the-art transformer approaches. Finally, we discuss the effects of progressive prompting strategies and the impact of augmented data in mitigating the biases manifested in IC-AnnoM. Our contributions provide the MI community with not only a comprehensive dataset but also valuable insights for using LLMs in empathetic text generation for conversational therapy in supervised settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12951v1">FineGates: LLMs Finetuning with Compression using Stochastic Gates</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), with billions of parameters, present significant challenges for full finetuning due to the high computational demands, memory requirements, and impracticality of many real-world applications. When faced with limited computational resources or small datasets, updating all model parameters can often result in overfitting. To address this, lightweight finetuning techniques have been proposed, like learning low-rank adapter layers. These methods aim to train only a few additional parameters combined with the base model, which remains frozen, reducing resource usage and mitigating overfitting risks. In this work, we propose an adaptor model based on stochastic gates that simultaneously sparsify the frozen base model with task-specific adaptation. Our method comes with a small number of trainable parameters and allows us to speed up the base model inference with competitive accuracy. We evaluate it in additional variants by equipping it with additional low-rank parameters and comparing it to several recent baselines. Our results show that the proposed method improves the finetuned model accuracy comparatively to the several baselines and allows the removal of up to 20-40\% without significant accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09916v2">ProxyLLM : LLM-Driven Framework for Customer Support Through Text-Style Transfer</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Chatbot-based customer support services have significantly advanced with the introduction of large language models (LLMs), enabling enhanced response quality and broader application across industries. However, while these advancements focus on reducing business costs and improving customer satisfaction, limited attention has been given to the experiences of customer service agents, who are critical to the service ecosystem. A major challenge faced by agents is the stress caused by unnecessary emotional exhaustion from harmful texts, which not only impairs their efficiency but also negatively affects customer satisfaction and business outcomes. In this work, we propose an LLM-powered system designed to enhance the working conditions of customer service agents by addressing emotionally intensive communications. Our proposed system leverages LLMs to transform the tone of customer messages, preserving actionable content while mitigating the emotional impact on human agents. Furthermore, the application is implemented as a Chrome extension, making it highly adaptable and easy to integrate into existing systems. Our method aims to enhance the overall service experience for businesses, customers, and agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11497v3">CrAM: Credibility-Aware Attention Modification in LLMs for Combating Misinformation in RAG</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 AAAI25 camera-ready
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) can alleviate hallucinations of Large Language Models (LLMs) by referencing external documents. However, the misinformation in external documents may mislead LLMs' generation. To address this issue, we explore the task of "credibility-aware RAG", in which LLMs automatically adjust the influence of retrieved documents based on their credibility scores to counteract misinformation. To this end, we introduce a plug-and-play method named $\textbf{Cr}$edibility-aware $\textbf{A}$ttention $\textbf{M}$odification (CrAM). CrAM identifies influential attention heads in LLMs and adjusts their attention weights based on the credibility of the documents, thereby reducing the impact of low-credibility documents. Experiments on Natual Questions and TriviaQA using Llama2-13B, Llama3-8B, and Qwen1.5-7B show that CrAM improves the RAG performance of LLMs against misinformation pollution by over 20%, even surpassing supervised fine-tuning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11491v2">SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Extended version of paper accepted to AAAI 2025. 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Safety alignment is indispensable for Large Language Models (LLMs) to defend threats from malicious instructions. However, recent researches reveal safety-aligned LLMs prone to reject benign queries due to the exaggerated safety issue, limiting their helpfulness. In this paper, we propose a Safety-Conscious Activation Steering (SCANS) method to mitigate the exaggerated safety concerns in aligned LLMs. First, SCANS extracts the refusal steering vectors within the activation space and utilizes vocabulary projection to anchor some specific safety-critical layers which influence model refusal behavior. Second, by tracking the hidden state transition, SCANS identifies the steering direction and steers the model behavior accordingly, achieving a balance between exaggerated safety and adequate safety. Experiments show that SCANS achieves new state-of-the-art performance on XSTest and OKTest benchmarks, without impairing their defense capability against harmful queries and maintaining almost unchanged model capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10033v3">Can GPT-O1 Kill All Bugs? An Evaluation of GPT-Family LLMs on QuixBugs</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 Accepted to the 6th International Workshop on Automated Program Repair (APR 2025)
    </div>
    <details class="paper-abstract">
      LLMs have long demonstrated remarkable effectiveness in automatic program repair (APR), with OpenAI's ChatGPT being one of the most widely used models in this domain. Through continuous iterations and upgrades of GPT-family models, their performance in fixing bugs has already reached state-of-the-art levels. However, there are few works comparing the effectiveness and variations of different versions of GPT-family models on APR. In this work, inspired by the recent public release of the GPT-o1 models, we conduct the first study to compare the effectiveness of different versions of the GPT-family models in APR. We evaluate the performance of the latest version of the GPT-family models (i.e., O1-preview and O1-mini), GPT-4o, and the historical version of ChatGPT on APR. We conduct an empirical study of the four GPT-family models against other LLMs and APR techniques on the QuixBugs benchmark from multiple evaluation perspectives, including repair success rate, repair cost, response length, and behavior patterns. The results demonstrate that O1's repair capability exceeds that of prior GPT-family models, successfully fixing all 40 bugs in the benchmark. Our work can serve as a foundation for further in-depth exploration of the applications of GPT-family models in APR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17679v3">Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-17
    </div>
    <details class="paper-abstract">
      Tokenization methods like Byte-Pair Encoding (BPE) enhance computational efficiency in large language models (LLMs) but often obscure internal character structures within tokens. This limitation hinders LLMs' ability to predict precise character positions, which is crucial in tasks like Chinese Spelling Correction (CSC) where identifying the positions of misspelled characters accelerates correction processes. We propose Token Internal Position Awareness (TIPA), a method that significantly improves models' ability to capture character positions within tokens by training them on reverse character prediction tasks using the tokenizer's vocabulary. Experiments demonstrate that TIPA enhances position prediction accuracy in LLMs, enabling more precise identification of target characters in original text. Furthermore, when applied to downstream tasks that do not require exact position prediction, TIPA still boosts performance in tasks needing character-level information, validating its versatility and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09056v3">Towards Reliable Detection of LLM-Generated Texts: A Comprehensive Evaluation Framework with CUDRT</a></div>
    <div class="paper-meta">
      📅 2024-12-17
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      The increasing prevalence of large language models (LLMs) has significantly advanced text generation, but the human-like quality of LLM outputs presents major challenges in reliably distinguishing between human-authored and LLM-generated texts. Existing detection benchmarks are constrained by their reliance on static datasets, scenario-specific tasks (e.g., question answering and text refinement), and a primary focus on English, overlooking the diverse linguistic and operational subtleties of LLMs. To address these gaps, we propose CUDRT, a comprehensive evaluation framework and bilingual benchmark in Chinese and English, categorizing LLM activities into five key operations: Create, Update, Delete, Rewrite, and Translate. CUDRT provides extensive datasets tailored to each operation, featuring outputs from state-of-the-art LLMs to assess the reliability of LLM-generated text detectors. This framework supports scalable, reproducible experiments and enables in-depth analysis of how operational diversity, multilingual training sets, and LLM architectures influence detection performance. Our extensive experiments demonstrate the framework's capacity to optimize detection systems, providing critical insights to enhance reliability, cross-linguistic adaptability, and detection accuracy. By advancing robust methodologies for identifying LLM-generated texts, this work contributes to the development of intelligent systems capable of meeting real-world multilingual detection challenges. Source code and dataset are available at GitHub.
    </details>
</div>
