# llm - 2024_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16807v2">Detecting LLM-Assisted Writing in Scientific Communication: Are We There Yet?</a></div>
    <div class="paper-meta">
      📅 2024-07-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), exemplified by ChatGPT, have significantly reshaped text generation, particularly in the realm of writing assistance. While ethical considerations underscore the importance of transparently acknowledging LLM use, especially in scientific communication, genuine acknowledgment remains infrequent. A potential avenue to encourage accurate acknowledging of LLM-assisted writing involves employing automated detectors. Our evaluation of four cutting-edge LLM-generated text detectors reveals their suboptimal performance compared to a simple ad-hoc detector designed to identify abrupt writing style changes around the time of LLM proliferation. We contend that the development of specialized detectors exclusively dedicated to LLM-assisted writing detection is necessary. Such detectors could play a crucial role in fostering more authentic recognition of LLM involvement in scientific communication, addressing the current challenges in acknowledgment practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04434v1">From 'Showgirls' to 'Performers': Fine-tuning with Gender-inclusive Language for Bias Reduction in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-05
      | 💬 10 pages, 5 tables; to appear in Proceedings of the 5th Workshop on Gender Bias in Natural Language Processing at ACL 2024
    </div>
    <details class="paper-abstract">
      Gender bias is not only prevalent in Large Language Models (LLMs) and their training data, but also firmly ingrained into the structural aspects of language itself. Therefore, adapting linguistic structures within LLM training data to promote gender-inclusivity can make gender representations within the model more inclusive. The focus of our work are gender-exclusive affixes in English, such as in 'show-girl' or 'man-cave', which can perpetuate gender stereotypes and binary conceptions of gender. We use an LLM training dataset to compile a catalogue of 692 gender-exclusive terms along with gender-neutral variants and from this, develop a gender-inclusive fine-tuning dataset, the 'Tiny Heap'. Fine-tuning three different LLMs with this dataset, we observe an overall reduction in gender-stereotyping tendencies across the models. Our approach provides a practical method for enhancing gender inclusivity in LLM training data and contributes to incorporating queer-feminist linguistic activism in bias mitigation research in NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04279v1">BiosERC: Integrating Biography Speakers Supported by LLMs for ERC Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-05
      | 💬 Accepted in the 33rd International Conference on Artificial Neural Networks (ICANN 2024)
    </div>
    <details class="paper-abstract">
      In the Emotion Recognition in Conversation task, recent investigations have utilized attention mechanisms exploring relationships among utterances from intra- and inter-speakers for modeling emotional interaction between them. However, attributes such as speaker personality traits remain unexplored and present challenges in terms of their applicability to other tasks or compatibility with diverse model architectures. Therefore, this work introduces a novel framework named BiosERC, which investigates speaker characteristics in a conversation. By employing Large Language Models (LLMs), we extract the "biographical information" of the speaker within a conversation as supplementary knowledge injected into the model to classify emotional labels for each utterance. Our proposed method achieved state-of-the-art (SOTA) results on three famous benchmark datasets: IEMOCAP, MELD, and EmoryNLP, demonstrating the effectiveness and generalization of our model and showcasing its potential for adaptation to various conversation analysis tasks. Our source code is available at https://github.com/yingjie7/BiosERC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10426v2">DELL: Generating Reactions and Explanations for LLM-Based Misinformation Detection</a></div>
    <div class="paper-meta">
      📅 2024-07-05
    </div>
    <details class="paper-abstract">
      Large language models are limited by challenges in factuality and hallucinations to be directly employed off-the-shelf for judging the veracity of news articles, where factual accuracy is paramount. In this work, we propose DELL that identifies three key stages in misinformation detection where LLMs could be incorporated as part of the pipeline: 1) LLMs could \emph{generate news reactions} to represent diverse perspectives and simulate user-news interaction networks; 2) LLMs could \emph{generate explanations} for proxy tasks (e.g., sentiment, stance) to enrich the contexts of news articles and produce experts specializing in various aspects of news understanding; 3) LLMs could \emph{merge task-specific experts} and provide an overall prediction by incorporating the predictions and confidence scores of varying experts. Extensive experiments on seven datasets with three LLMs demonstrate that DELL outperforms state-of-the-art baselines by up to 16.8\% in macro f1-score. Further analysis reveals that the generated reactions and explanations are greatly helpful in misinformation detection, while our proposed LLM-guided expert merging helps produce better-calibrated predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04181v1">Orchestrating LLMs with Different Personalizations</a></div>
    <div class="paper-meta">
      📅 2024-07-04
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to aligning large language models (LLMs) with individual human preferences, sometimes referred to as Reinforcement Learning from \textit{Personalized} Human Feedback (RLPHF). Given stated preferences along multiple dimensions, such as helpfulness, conciseness, or humor, the goal is to create an LLM without re-training that best adheres to this specification. Starting from specialized expert LLMs, each trained for one such particular preference dimension, we propose a black-box method that merges their outputs on a per-token level. We train a lightweight Preference Control Model (PCM) that dynamically translates the preference description and current context into next-token prediction weights. By combining the expert models' outputs at the token level, our approach dynamically generates text that optimizes the given preference. Empirical tests show that our method matches or surpasses existing preference merging techniques, providing a scalable, efficient alternative to fine-tuning LLMs for individual personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18328v1">Unveiling Scoring Processes: Dissecting the Differences between LLMs and Human Graders in Automatic Scoring</a></div>
    <div class="paper-meta">
      📅 2024-07-04
      | 💬 Non-archival Presenting at EDM 2024 Workshop on Large Language Models
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong potential in performing automatic scoring for constructed response assessments. While constructed responses graded by humans are usually based on given grading rubrics, the methods by which LLMs assign scores remain largely unclear. It is also uncertain how closely AI's scoring process mirrors that of humans, or if it adheres to the same grading criteria. To address this gap, this paper uncovers the grading rubrics that LLMs used to score students' written responses to science tasks and their alignment with human scores. We also examine whether enhancing the alignments can improve scoring accuracy. Specifically, we prompt LLMs to generate analytic rubrics that they use to assign scores and study the alignment gap with human grading rubrics. Based on a series of experiments with various configurations of LLM settings, we reveal a notable alignment gap between human and LLM graders. While LLMs can adapt quickly to scoring tasks, they often resort to shortcuts, bypassing deeper logical reasoning expected in human grading. We found that incorporating high-quality analytical rubrics designed to reflect human grading logic can mitigate this gap and enhance LLMs' scoring accuracy. These results caution against the simplistic application of LLMs in science education and highlight the importance of aligning LLM outputs with human expectations to ensure efficient and accurate automatic scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04173v1">Quantifying Prediction Consistency Under Model Multiplicity in Tabular LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-04
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on limited tabular data for classification tasks can lead to \textit{fine-tuning multiplicity}, where equally well-performing models make conflicting predictions on the same inputs due to variations in the training process (i.e., seed, random weight initialization, retraining on additional or deleted samples). This raises critical concerns about the robustness and reliability of Tabular LLMs, particularly when deployed for high-stakes decision-making, such as finance, hiring, education, healthcare, etc. This work formalizes the challenge of fine-tuning multiplicity in Tabular LLMs and proposes a novel metric to quantify the robustness of individual predictions without expensive model retraining. Our metric quantifies a prediction's stability by analyzing (sampling) the model's local behavior around the input in the embedding space. Interestingly, we show that sampling in the local neighborhood can be leveraged to provide probabilistic robustness guarantees against a broad class of fine-tuned models. By leveraging Bernstein's Inequality, we show that predictions with sufficiently high robustness (as defined by our measure) will remain consistent with high probability. We also provide empirical evaluation on real-world datasets to support our theoretical results. Our work highlights the importance of addressing fine-tuning instabilities to enable trustworthy deployment of LLMs in high-stakes and safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04067v1">Semantic Graphs for Syntactic Simplification: A Revisit from the Age of LLM</a></div>
    <div class="paper-meta">
      📅 2024-07-04
      | 💬 Accepted at TextGraphs-17 @ ACL 2024
    </div>
    <details class="paper-abstract">
      Symbolic sentence meaning representations, such as AMR (Abstract Meaning Representation) provide expressive and structured semantic graphs that act as intermediates that simplify downstream NLP tasks. However, the instruction-following capability of large language models (LLMs) offers a shortcut to effectively solve NLP tasks, questioning the utility of semantic graphs. Meanwhile, recent work has also shown the difficulty of using meaning representations merely as a helpful auxiliary for LLMs. We revisit the position of semantic graphs in syntactic simplification, the task of simplifying sentence structures while preserving their meaning, which requires semantic understanding, and evaluate it on a new complex and natural dataset. The AMR-based method that we propose, AMRS$^3$, demonstrates that state-of-the-art meaning representations can lead to easy-to-implement simplification methods with competitive performance and unique advantages in cost, interpretability, and generalization. With AMRS$^3$ as an anchor, we discover that syntactic simplification is a task where semantic graphs are helpful in LLM prompting. We propose AMRCoC prompting that guides LLMs to emulate graph algorithms for explicit symbolic reasoning on AMR graphs, and show its potential for improving LLM on semantic-centered tasks like syntactic simplification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04046v1">Systematic Task Exploration with LLMs: A Study in Citation Text Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-04
      | 💬 Accepted to ACL 2024 (Main)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) bring unprecedented flexibility in defining and executing complex, creative natural language generation (NLG) tasks. Yet, this flexibility brings new challenges, as it introduces new degrees of freedom in formulating the task inputs and instructions and in evaluating model performance. To facilitate the exploration of creative NLG tasks, we propose a three-component research framework that consists of systematic input manipulation, reference data, and output measurement. We use this framework to explore citation text generation -- a popular scholarly NLP task that lacks consensus on the task definition and evaluation metric and has not yet been tackled within the LLM paradigm. Our results highlight the importance of systematically investigating both task instruction and input configuration when prompting LLMs, and reveal non-trivial relationships between different evaluation metrics used for citation text generation. Additional human generation and human evaluation experiments provide new qualitative insights into the task to guide future research in citation text generation. We make our code and data publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04014v1">Offline Energy-Optimal LLM Serving: Workload-Based Energy Models for LLM Inference on Heterogeneous Systems</a></div>
    <div class="paper-meta">
      📅 2024-07-04
      | 💬 7 pages, appearing at HotCarbon 2024
    </div>
    <details class="paper-abstract">
      The rapid adoption of large language models (LLMs) has led to significant advances in natural language processing and text generation. However, the energy consumed through LLM model inference remains a major challenge for sustainable AI deployment. To address this problem, we model the workload-dependent energy consumption and runtime of LLM inference tasks on heterogeneous GPU-CPU systems. By conducting an extensive characterization study of several state-of-the-art LLMs and analyzing their energy and runtime behavior across different magnitudes of input prompts and output text, we develop accurate (R^2>0.96) energy and runtime models for each LLM. We employ these models to explore an offline, energy-optimal LLM workload scheduling framework. Through a case study, we demonstrate the advantages of energy and accuracy aware scheduling compared to existing best practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02669v2">Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache</a></div>
    <div class="paper-meta">
      📅 2024-07-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate substantial potential across a diverse array of domains via request serving. However, as trends continue to push for expanding context sizes, the autoregressive nature of LLMs results in highly dynamic behavior of the attention layers, showcasing significant differences in computational characteristics and memory requirements from the non-attention layers. This presents substantial challenges for resource management and performance optimization in service systems. Existing static model parallelism and resource allocation strategies fall short when dealing with this dynamicity. To address the issue, we propose Infinite-LLM, a novel LLM serving system designed to effectively handle dynamic context lengths. Infinite-LLM disaggregates attention layers from an LLM's inference process, facilitating flexible and independent resource scheduling that optimizes computational performance and enhances memory utilization jointly. By leveraging a pooled GPU memory strategy across a cluster, Infinite-LLM not only significantly boosts system throughput but also supports extensive context lengths. Evaluated on a dataset with context lengths ranging from a few to 2000K tokens across a cluster with 32 A100 GPUs, Infinite-LLM demonstrates throughput improvement of 1.35-3.4x compared to state-of-the-art methods, enabling efficient and elastic LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14016v2">Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment</a></div>
    <div class="paper-meta">
      📅 2024-07-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful zero-shot assessors used in real-world situations such as assessing written exams and benchmarking systems. Despite these critical applications, no existing work has analyzed the vulnerability of judge-LLMs to adversarial manipulation. This work presents the first study on the adversarial robustness of assessment LLMs, where we demonstrate that short universal adversarial phrases can be concatenated to deceive judge LLMs to predict inflated scores. Since adversaries may not know or have access to the judge-LLMs, we propose a simple surrogate attack where a surrogate model is first attacked, and the learned attack phrase then transferred to unknown judge-LLMs. We propose a practical algorithm to determine the short universal attack phrases and demonstrate that when transferred to unseen models, scores can be drastically inflated such that irrespective of the assessed text, maximum scores are predicted. It is found that judge-LLMs are significantly more susceptible to these adversarial attacks when used for absolute scoring, as opposed to comparative assessment. Our findings raise concerns on the reliability of LLM-as-a-judge methods, and emphasize the importance of addressing vulnerabilities in LLM assessment methods before deployment in high-stakes real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03841v1">On the Benchmarking of LLMs for Open-Domain Dialogue Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-07-04
      | 💬 Accepted to the 6th NLP for Conversational AI workshop at ACL
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have showcased remarkable capabilities in various Natural Language Processing tasks. For automatic open-domain dialogue evaluation in particular, LLMs have been seamlessly integrated into evaluation frameworks, and together with human evaluation, compose the backbone of most evaluations. However, existing evaluation benchmarks often rely on outdated datasets and evaluate aspects like Fluency and Relevance, which fail to adequately capture the capabilities and limitations of state-of-the-art chatbot models. This paper critically examines current evaluation benchmarks, highlighting that the use of older response generators and quality aspects fail to accurately reflect modern chatbot capabilities. A small annotation experiment on a recent LLM-generated dataset (SODA) reveals that LLM evaluators such as GPT-4 struggle to detect actual deficiencies in dialogues generated by current LLM chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06341v2">RareBench: Can LLMs Serve as Rare Diseases Specialists?</a></div>
    <div class="paper-meta">
      📅 2024-07-04
      | 💬 KDD2024
    </div>
    <details class="paper-abstract">
      Generalist Large Language Models (LLMs), such as GPT-4, have shown considerable promise in various domains, including medical diagnosis. Rare diseases, affecting approximately 300 million people worldwide, often have unsatisfactory clinical diagnosis rates primarily due to a lack of experienced physicians and the complexity of differentiating among many rare diseases. In this context, recent news such as "ChatGPT correctly diagnosed a 4-year-old's rare disease after 17 doctors failed" underscore LLMs' potential, yet underexplored, role in clinically diagnosing rare diseases. To bridge this research gap, we introduce RareBench, a pioneering benchmark designed to systematically evaluate the capabilities of LLMs on 4 critical dimensions within the realm of rare diseases. Meanwhile, we have compiled the largest open-source dataset on rare disease patients, establishing a benchmark for future studies in this domain. To facilitate differential diagnosis of rare diseases, we develop a dynamic few-shot prompt methodology, leveraging a comprehensive rare disease knowledge graph synthesized from multiple knowledge bases, significantly enhancing LLMs' diagnostic performance. Moreover, we present an exhaustive comparative study of GPT-4's diagnostic capabilities against those of specialist physicians. Our experimental findings underscore the promising potential of integrating LLMs into the clinical diagnostic process for rare diseases. This paves the way for exciting possibilities in future advancements in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18326v1">Classification-Based Automatic HDL Code Generation Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-04
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated the ability to generate hardware description language (HDL) code for digital circuits, they still suffer from the hallucination problem, which leads to the generation of incorrect HDL code or misunderstanding of specifications. In this work, we introduce a human-expert-inspired method to mitigate the hallucination of LLMs and improve the performance in HDL code generation. We first let LLMs classify the type of the circuit based on the specifications. Then, according to the type of the circuit, we split the tasks into several sub-procedures, including information extraction and human-like design flow using Electronic Design Automation (EDA) tools. Besides, we also use a search method to mitigate the variation in code generation. Experimental results show that our method can significantly improve the functional correctness of the generated Verilog and reduce the hallucination of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03678v1">Improving Self Consistency in LLMs through Probabilistic Tokenization</a></div>
    <div class="paper-meta">
      📅 2024-07-04
      | 💬 ICML 2024 Workshop on LLMs and Cognition
    </div>
    <details class="paper-abstract">
      Prior research has demonstrated noticeable performance gains through the use of probabilistic tokenizations, an approach that involves employing multiple tokenizations of the same input string during the training phase of a language model. Despite these promising findings, modern large language models (LLMs) have yet to be trained using probabilistic tokenizations. Interestingly, while the tokenizers of these contemporary LLMs have the capability to generate multiple tokenizations, this property remains underutilized. In this work, we propose a novel method to leverage the multiple tokenization capabilities of modern LLM tokenizers, aiming to enhance the self-consistency of LLMs in reasoning tasks. Our experiments indicate that when utilizing probabilistic tokenizations, LLMs generate logically diverse reasoning paths, moving beyond mere surface-level linguistic diversity.We carefully study probabilistic tokenization and offer insights to explain the self consistency improvements it brings through extensive experimentation on 5 LLM families and 4 reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03855v4">PythonSaga: Redefining the Benchmark to Evaluate Code Generating LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-04
    </div>
    <details class="paper-abstract">
      Driven by the surge in code generation using large language models (LLMs), numerous benchmarks have emerged to evaluate these LLMs capabilities. We conducted a large-scale human evaluation of HumanEval and MBPP, two popular benchmarks for Python code generation, analyzing their diversity and difficulty. Our findings unveil a critical bias towards a limited set of programming concepts, neglecting most of the other concepts entirely. Furthermore, we uncover a worrying prevalence of easy tasks, potentially inflating model performance estimations. To address these limitations, we propose a novel benchmark, PythonSaga, featuring 185 hand-crafted prompts on a balanced representation of 38 programming concepts across diverse difficulty levels. The robustness of our benchmark is demonstrated by the poor performance of existing Code-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03568v1">When LLM Meets Hypergraph: A Sociological Analysis on Personality via Online Social Networks</a></div>
    <div class="paper-meta">
      📅 2024-07-04
    </div>
    <details class="paper-abstract">
      Individual personalities significantly influence our perceptions, decisions, and social interactions, which is particularly crucial for gaining insights into human behavior patterns in online social network analysis. Many psychological studies have observed that personalities are strongly reflected in their social behaviors and social environments. In light of these problems, this paper proposes a sociological analysis framework for one's personality in an environment-based view instead of individual-level data mining. Specifically, to comprehensively understand an individual's behavior from low-quality records, we leverage the powerful associative ability of LLMs by designing an effective prompt. In this way, LLMs can integrate various scattered information with their external knowledge to generate higher-quality profiles, which can significantly improve the personality analysis performance. To explore the interactive mechanism behind the users and their online environments, we design an effective hypergraph neural network where the hypergraph nodes are users and the hyperedges in the hypergraph are social environments. We offer a useful dataset with user profile data, personality traits, and several detected environments from the real-world social platform. To the best of our knowledge, this is the first network-based dataset containing both hypergraph structure and social information, which could push forward future research in this area further. By employing the framework on this dataset, we can effectively capture the nuances of individual personalities and their online behaviors, leading to a deeper understanding of human interactions in the digital world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.04444v4">What's the Magic Word? A Control Theory of LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 28 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Prompt engineering is crucial for deploying LLMs but is poorly understood mathematically. We formalize LLM systems as a class of discrete stochastic dynamical systems to explore prompt engineering through the lens of control theory. We offer a mathematical analysis of the limitations on the controllability of self-attention as a function of the singular values of the parameter matrices. We present complementary empirical results on the controllability of a panel of LLMs, including Falcon-7b, Llama-7b, and Falcon-40b. Given initial state $\mathbf x_0$ from Wikitext and prompts of length $k \leq 10$ tokens, we find that the "correct" next token is reachable at least 97% of the time, and that the top 75 most likely next tokens are reachable at least 85% of the time. Intriguingly, short prompt sequences can dramatically alter the likelihood of specific outputs, even making the least likely tokens become the most likely ones. This control-theoretic analysis of LLMs demonstrates the significant and poorly understood role of input sequences in steering output probabilities, offering a foundational perspective for enhancing language model system capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03479v1">Human-Centered Design Recommendations for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 14 pages, 6 figures, Accepted for publication in ACL 2024 Workshop HuCLLM
    </div>
    <details class="paper-abstract">
      Traditional reference-based metrics, such as BLEU and ROUGE, are less effective for assessing outputs from Large Language Models (LLMs) that produce highly creative or superior-quality text, or in situations where reference outputs are unavailable. While human evaluation remains an option, it is costly and difficult to scale. Recent work using LLMs as evaluators (LLM-as-a-judge) is promising, but trust and reliability remain a significant concern. Integrating human input is crucial to ensure criteria used to evaluate are aligned with the human's intent, and evaluations are robust and consistent. This paper presents a user study of a design exploration called EvaluLLM, that enables users to leverage LLMs as customizable judges, promoting human involvement to balance trust and cost-saving potential with caution. Through interviews with eight domain experts, we identified the need for assistance in developing effective evaluation criteria aligning the LLM-as-a-judge with practitioners' preferences and expectations. We offer findings and design recommendations to optimize human-assisted LLM-as-judge systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03460v1">Collaborative Quest Completion with LLM-driven Non-Player Characters in Minecraft</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 Accepted at Wordplay workshop at ACL 2024
    </div>
    <details class="paper-abstract">
      The use of generative AI in video game development is on the rise, and as the conversational and other capabilities of large language models continue to improve, we expect LLM-driven non-player characters (NPCs) to become widely deployed. In this paper, we seek to understand how human players collaborate with LLM-driven NPCs to accomplish in-game goals. We design a minigame within Minecraft where a player works with two GPT4-driven NPCs to complete a quest. We perform a user study in which 28 Minecraft players play this minigame and share their feedback. On analyzing the game logs and recordings, we find that several patterns of collaborative behavior emerge from the NPCs and the human players. We also report on the current limitations of language-only models that do not have rich game-state or visual understanding. We believe that this preliminary study and analysis will inform future game developers on how to better exploit these rapidly improving generative AI models for collaborative roles in games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11030v1">DLO: Dynamic Layer Operation for Efficient Vertical Scaling of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-03
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Dynamic Layer Operations (DLO), a novel approach for vertically scaling transformer-based Large Language Models (LLMs) by dynamically expanding, activating, or skipping layers using a sophisticated routing policy based on layerwise feature similarity. Unlike traditional Mixture-of-Experts (MoE) methods that focus on extending the model width, our approach targets model depth, addressing the redundancy observed across layer representations for various input samples. Our framework is integrated with the Supervised Fine-Tuning (SFT) stage, eliminating the need for resource-intensive Continual Pre-Training (CPT). Experimental results demonstrate that DLO not only outperforms the original unscaled models but also achieves comparable results to densely expanded models with significantly improved efficiency. Our work offers a promising direction for building efficient yet powerful LLMs. We will release our implementation and model weights upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00589v2">Merlin:Empowering Multimodal LLMs with Foresight Minds</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 Accepted by ECCV2024. Project page: https://ahnsun.github.io/merlin
    </div>
    <details class="paper-abstract">
      Humans possess the remarkable ability to foresee the future to a certain extent based on present observations, a skill we term as foresight minds. However, this capability remains largely under explored within existing Multimodal Large Language Models (MLLMs), hindering their capacity to learn the fundamental principles of how things operate and the intentions behind the observed subjects. To address this issue, we introduce the integration of future modeling into the existing learning frameworks of MLLMs. By utilizing the subject trajectory, a highly structured representation of a consecutive frame sequence, as a learning objective, we aim to bridge the gap between the past and the future. We propose two innovative methods to empower MLLMs with foresight minds, Foresight Pre-Training (FPT) and Foresight Instruction-Tuning (FIT), which are inspired by the modern learning paradigm of LLMs. Specifically, FPT jointly training various tasks centered on trajectories, enabling MLLMs to learn how to attend and predict entire trajectories from a given initial observation. Then, FIT requires MLLMs to first predict trajectories of related objects and then reason about potential future events based on them. Aided by FPT and FIT, we build a novel and unified MLLM named Merlin that supports multi-images input and analysis about potential actions of multiple objects for the future reasoning. Experimental results show Merlin powerful foresight minds with impressive performance on both future reasoning and visual comprehension tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03232v1">Single Character Perturbations Break LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      When LLMs are deployed in sensitive, human-facing settings, it is crucial that they do not output unsafe, biased, or privacy-violating outputs. For this reason, models are both trained and instructed to refuse to answer unsafe prompts such as "Tell me how to build a bomb." We find that, despite these safeguards, it is possible to break model defenses simply by appending a space to the end of a model's input. In a study of eight open-source models, we demonstrate that this acts as a strong enough attack to cause the majority of models to generate harmful outputs with very high success rates. We examine the causes of this behavior, finding that the contexts in which single spaces occur in tokenized training data encourage models to generate lists when prompted, overriding training signals to refuse to answer unsafe requests. Our findings underscore the fragile state of current model alignment and promote the importance of developing more robust alignment methods. Code and data will be available at https://github.com/hannah-aught/space_attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08593v2">Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 Accepted by ACL 2024 Findings. 21 pages, 7 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown potential in reasoning over structured environments, e.g., knowledge graph and table. Such tasks typically require multi-hop reasoning, i.e., match natural language utterance with instances in the environment. Previous methods leverage LLMs to incrementally build a reasoning path, where the LLMs either invoke tools or pick up schemas by step-by-step interacting with the environment. We propose Reasoning-Path-Editing (Readi), a novel framework where LLMs can efficiently and faithfully reason over structured environments. In Readi, LLMs initially generate a reasoning path given a query, and edit the path only when necessary. We instantiate the path on structured environments and provide feedback to edit the path if anything goes wrong. Experimental results on three KGQA and two TableQA datasets show the effectiveness of Readi, significantly surpassing previous LLM-based methods (by 9.1% Hit@1 on WebQSP, 12.4% on MQA-3H and 9.5% on WTQ), comparable with state-of-the-art fine-tuned methods (67% on CWQ and 74.7% on WebQSP) and substantially boosting the vanilla LLMs (by 14.9% on CWQ). Our code will be available on https://aka.ms/readi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01910v2">MG-Verilog: Multi-grained Dataset Towards Enhanced LLM-assisted Verilog Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 Accepted in ISLAD 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown promise in streamlining hardware design processes by encapsulating vast amounts of domain-specific data. In addition, they allow users to interact with the design processes through natural language instructions, thus making hardware design more accessible to developers. However, effectively leveraging LLMs in hardware design necessitates providing domain-specific data during inference (e.g., through in-context learning), fine-tuning, or pre-training. Unfortunately, existing publicly available hardware datasets are often limited in size, complexity, or detail, which hinders the effectiveness of LLMs in hardware design tasks. To address this issue, we first propose a set of criteria for creating high-quality hardware datasets that can effectively enhance LLM-assisted hardware design. Based on these criteria, we propose a Multi-Grained-Verilog (MG-Verilog) dataset, which encompasses descriptions at various levels of detail and corresponding code samples. To benefit the broader hardware design community, we have developed an open-source infrastructure that facilitates easy access, integration, and extension of the dataset to meet specific project needs. Furthermore, to fully exploit the potential of the MG-Verilog dataset, which varies in complexity and detail, we introduce a balanced fine-tuning scheme. This scheme serves as a unique use case to leverage the diverse levels of detail provided by the dataset. Extensive experiments demonstrate that the proposed dataset and fine-tuning scheme consistently improve the performance of LLMs in hardware design tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03391v1">Soft Begging: Modular and Efficient Shielding of LLMs against Prompt Injection and Jailbreaking based on Prompt Tuning</a></div>
    <div class="paper-meta">
      📅 2024-07-03
    </div>
    <details class="paper-abstract">
      Prompt injection (both direct and indirect) and jailbreaking are now recognized as significant issues for large language models (LLMs), particularly due to their potential for harm in application-integrated contexts. This extended abstract explores a novel approach to protecting LLMs from such attacks, termed "soft begging." This method involves training soft prompts to counteract the effects of corrupted prompts on the LLM's output. We provide an overview of prompt injections and jailbreaking, introduce the theoretical basis of the "soft begging" technique, and discuss an evaluation of its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13022v2">LLMs can learn self-restraint through iterative self-reflection</a></div>
    <div class="paper-meta">
      📅 2024-07-03
    </div>
    <details class="paper-abstract">
      In order to be deployed safely, Large Language Models (LLMs) must be capable of dynamically adapting their behavior based on their level of knowledge and uncertainty associated with specific topics. This adaptive behavior, which we refer to as self-restraint, is non-trivial to teach since it depends on the internal knowledge of an LLM. By default, LLMs are trained to maximize the next token likelihood, which does not teach the model to modulate its answer based on its level of uncertainty. In order to learn self-restraint, we devise a utility function that can encourage the model to produce responses only when it is confident in them. This utility function can be used to score generation of different length and abstention. To optimize this function, we introduce ReSearch, a process of "self-reflection" consisting of iterative self-prompting and self-evaluation. We use the ReSearch algorithm to generate synthetic data on which we finetune our models. Compared to their original versions, our resulting models generate fewer \emph{hallucinations} overall at no additional inference cost, for both known and unknown topics, as the model learns to selectively restrain itself. In addition, our method elegantly incorporates the ability to abstain by augmenting the samples generated by the model during the search procedure with an answer expressing abstention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03157v1">Let the Code LLM Edit Itself When You Edit the Code</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 Preprint. Work in Progress
    </div>
    <details class="paper-abstract">
      In this work, we investigate a typical scenario in code generation where a developer edits existing code in real time and requests a code assistant, e.g., a large language model, to re-predict the next token or next line on the fly. Naively, the LLM needs to re-encode the entire KV cache to provide an accurate prediction. However, this process is computationally expensive, especially when the sequence length is long. Simply encoding the edited subsequence and integrating it to the original KV cache meets the temporal confusion problem, leading to significantly worse performance. We address this efficiency and accuracy trade-off by introducing \underline{\textbf{Positional \textbf{I}ntegrity \textbf{E}ncoding} (PIE). Building upon the rotary positional encoding, PIE first removes the rotary matrices in the Key cache that introduce temporal confusion and then reapplies the correct rotary matrices. This process ensures that positional relationships between tokens are correct and requires only a single round of matrix multiplication. We validate the effectiveness of PIE through extensive experiments on the RepoBench-C-8k dataset, utilizing DeepSeek-Coder models with 1.3B, 6.7B, and 33B parameters. Our evaluation includes three real-world coding tasks: code insertion, code deletion, and multi-place code editing. Results demonstrate that PIE reduces computational overhead by over 85% compared to the standard full recomputation approach across all model sizes and tasks while well approximating the model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16203v3">LLMs' Classification Performance is Overclaimed</a></div>
    <div class="paper-meta">
      📅 2024-07-03
    </div>
    <details class="paper-abstract">
      In many classification tasks designed for AI or human to solve, gold labels are typically included within the label space by default, often posed as "which of the following is correct?" This standard setup has traditionally highlighted the strong performance of advanced AI, particularly top-performing Large Language Models (LLMs), in routine classification tasks. However, when the gold label is intentionally excluded from the label space, it becomes evident that LLMs still attempt to select from the available label candidates, even when none are correct. This raises a pivotal question: Do LLMs truly demonstrate their intelligence in understanding the essence of classification tasks? In this study, we evaluate both closed-source and open-source LLMs across representative classification tasks, arguing that the perceived performance of LLMs is overstated due to their inability to exhibit the expected comprehension of the task. This paper makes a threefold contribution: i) To our knowledge, this is the first work to identify the limitations of LLMs in classification tasks when gold labels are absent. We define this task as Classify-w/o-Gold and propose it as a new testbed for LLMs. ii) We introduce a benchmark, Know-No, comprising two existing classification tasks and one new task, to evaluate Classify-w/o-Gold. iii) This work defines and advocates for a new evaluation metric, OmniAccuracy, which assesses LLMs' performance in classification tasks both when gold labels are present and absent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03045v1">JailbreakHunter: A Visual Analytics Approach for Jailbreak Prompts Discovery from Large-Scale Human-LLM Conversational Datasets</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 18 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained significant attention but also raised concerns due to the risk of misuse. Jailbreak prompts, a popular type of adversarial attack towards LLMs, have appeared and constantly evolved to breach the safety protocols of LLMs. To address this issue, LLMs are regularly updated with safety patches based on reported jailbreak prompts. However, malicious users often keep their successful jailbreak prompts private to exploit LLMs. To uncover these private jailbreak prompts, extensive analysis of large-scale conversational datasets is necessary to identify prompts that still manage to bypass the system's defenses. This task is highly challenging due to the immense volume of conversation data, diverse characteristics of jailbreak prompts, and their presence in complex multi-turn conversations. To tackle these challenges, we introduce JailbreakHunter, a visual analytics approach for identifying jailbreak prompts in large-scale human-LLM conversational datasets. We have designed a workflow with three analysis levels: group-level, conversation-level, and turn-level. Group-level analysis enables users to grasp the distribution of conversations and identify suspicious conversations using multiple criteria, such as similarity with reported jailbreak prompts in previous research and attack success rates. Conversation-level analysis facilitates the understanding of the progress of conversations and helps discover jailbreak prompts within their conversation contexts. Turn-level analysis allows users to explore the semantic similarity and token overlap between a singleturn prompt and the reported jailbreak prompts, aiding in the identification of new jailbreak strategies. The effectiveness and usability of the system were verified through multiple case studies and expert interviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09834v2">How and Why LLMs Use Deprecated APIs in Code Completion? An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2024-07-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), pre-trained or fine-tuned on large code corpora, have shown effectiveness in generating code completions. However, in LLM-based code completion, LLMs may struggle to use correct and up-to-date Application Programming Interfaces (APIs) due to the rapid and continuous evolution of libraries. While existing studies have highlighted issues with predicting incorrect APIs, the specific problem of deprecated API usage in LLM-based code completion has not been thoroughly investigated. To address this gap, we conducted the first evaluation study on deprecated API usage in LLM-based code completion. This study involved seven advanced LLMs, 145 API mappings from eight popular Python libraries, and 28,125 completion prompts. The study results reveal the \textit{status quo} and \textit{root causes} of deprecated API usage in LLM-based code completion from the perspectives of \textit{model}, \textit{prompt}, and \textit{library}. Based on these findings, we propose two lightweight fixing approaches, \textsc{ReplaceAPI} and \textsc{InsertPrompt}, which can serve as baseline approaches for future research on mitigating deprecated API usage in LLM-based completion. Additionally, we provide implications for future research on integrating library evolution with LLM-driven software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02824v1">Exploring the Capabilities of LLMs for Code Change Related Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-03
    </div>
    <details class="paper-abstract">
      Developers deal with code-change-related tasks daily, e.g., reviewing code. Pre-trained code and code-change-oriented models have been adapted to help developers with such tasks. Recently, large language models (LLMs) have shown their effectiveness in code-related tasks. However, existing LLMs for code focus on general code syntax and semantics rather than the differences between two code versions. Thus, it is an open question how LLMs perform on code-change-related tasks. To answer this question, we conduct an empirical study using \textgreater 1B parameters LLMs on three code-change-related tasks, i.e., code review generation, commit message generation, and just-in-time comment update, with in-context learning (ICL) and parameter-efficient fine-tuning (PEFT, including LoRA and prefix-tuning). We observe that the performance of LLMs is poor without examples and generally improves with examples, but more examples do not always lead to better performance. LLMs tuned with LoRA have comparable performance to the state-of-the-art small pre-trained models. Larger models are not always better, but \textsc{Llama~2} and \textsc{Code~Llama} families are always the best. The best LLMs outperform small pre-trained models on the code changes that only modify comments and perform comparably on other code changes. We suggest future work should focus more on guiding LLMs to learn the knowledge specific to the changes related to code rather than comments for code-change-related tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10671v3">Decomposition for Enhancing Attention: Improving LLM-based Text-to-SQL through Workflow Paradigm</a></div>
    <div class="paper-meta">
      📅 2024-07-03
    </div>
    <details class="paper-abstract">
      In-context learning of large-language models (LLMs) has achieved remarkable success in the field of natural language processing, while extensive case studies reveal that the single-step chain-of-thought prompting approach faces challenges such as attention diffusion and inadequate performance in complex tasks like text-to-SQL. To improve the contextual learning capabilities of LLMs in text-to-SQL, a workflow paradigm method is proposed, aiming to enhance the attention and problem-solving scope of LLMs through decomposition. Specifically, the information determination module for eliminating redundant information and the brand-new prompt structure based on problem classification greatly enhance the model's attention. Additionally, the inclusion of self-correction and active learning modules greatly expands the problem-solving scope of LLMs, hence improving the upper limit of LLM-based approaches. Extensive experiments conducted on three datasets demonstrate that our approach outperforms other methods by a significant margin. About 2-3 percentage point improvements compared to the existing baseline on the Spider Dev, Spider-Realistic, and Bird Dev datasets and new SOTA results on the Spider Test dataset are achieved. Our code is available on GitHub: \url{https://github.com/FlyingFeather/DEA-SQL}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02411v2">Video Watermarking: Safeguarding Your Video from (Unauthorized) Annotations by Video-based LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 arXiv admin note: substantial text overlap with arXiv:2403.13507
    </div>
    <details class="paper-abstract">
      The advent of video-based Large Language Models (LLMs) has significantly enhanced video understanding. However, it has also raised some safety concerns regarding data protection, as videos can be more easily annotated, even without authorization. This paper introduces Video Watermarking, a novel technique to protect videos from unauthorized annotations by such video-based LLMs, especially concerning the video content and description, in response to specific queries. By imperceptibly embedding watermarks into key video frames with multi-modal flow-based losses, our method preserves the viewing experience while preventing misuse by video-based LLMs. Extensive experiments show that Video Watermarking significantly reduces the comprehensibility of videos with various video-based LLMs, demonstrating both stealth and robustness. In essence, our method provides a solution for securing video content, ensuring its integrity and confidentiality in the face of evolving video-based LLMs technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02791v1">Model-Enhanced LLM-Driven VUI Testing of VPA Apps</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The flourishing ecosystem centered around voice personal assistants (VPA), such as Amazon Alexa, has led to the booming of VPA apps. The largest app market Amazon skills store, for example, hosts over 200,000 apps. Despite their popularity, the open nature of app release and the easy accessibility of apps also raise significant concerns regarding security, privacy and quality. Consequently, various testing approaches have been proposed to systematically examine VPA app behaviors. To tackle the inherent lack of a visible user interface in the VPA app, two strategies are employed during testing, i.e., chatbot-style testing and model-based testing. The former often lacks effective guidance for expanding its search space, while the latter falls short in interpreting the semantics of conversations to construct precise and comprehensive behavior models for apps. In this work, we introduce Elevate, a model-enhanced large language model (LLM)-driven VUI testing framework. Elevate leverages LLMs' strong capability in natural language processing to compensate for semantic information loss during model-based VUI testing. It operates by prompting LLMs to extract states from VPA apps' outputs and generate context-related inputs. During the automatic interactions with the app, it incrementally constructs the behavior model, which facilitates the LLM in generating inputs that are highly likely to discover new states. Elevate bridges the LLM and the behavior model with innovative techniques such as encoding behavior model into prompts and selecting LLM-generated inputs based on the context relevance. Elevate is benchmarked on 4,000 real-world Alexa skills, against the state-of-the-art tester Vitas. It achieves 15% higher state space coverage compared to Vitas on all types of apps, and exhibits significant advancement in efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13210v2">Bayesian Reward Models for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-07-03
    </div>
    <details class="paper-abstract">
      To ensure that large language model (LLM) responses are helpful and non-toxic, a reward model trained on human preference data is usually used. LLM responses with high rewards are then selected through best-of-$n$ (BoN) sampling or the LLM is further optimized to produce responses with high rewards through reinforcement learning from human feedback (RLHF). However, these processes are susceptible to reward overoptimization or `hacking', where responses receive high rewards due to imperfections in the reward model rather than true preference, particularly as prompts or responses deviate from the training data. To address these challenges, we propose to train a Bayesian reward model, which signals higher uncertainty further from the training data distribution. We trained Bayesian reward models using Laplace approximation on LoRA weights, and found that the resulting uncertainty estimates can effectively mitigate reward overoptimization in BoN sampling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02662v1">Supporters and Skeptics: LLM-based Analysis of Engagement with Mental Health (Mis)Information Content on Video-sharing Platforms</a></div>
    <div class="paper-meta">
      📅 2024-07-02
      | 💬 12 pages, in submission to ICWSM
    </div>
    <details class="paper-abstract">
      Over one in five adults in the US lives with a mental illness. In the face of a shortage of mental health professionals and offline resources, online short-form video content has grown to serve as a crucial conduit for disseminating mental health help and resources. However, the ease of content creation and access also contributes to the spread of misinformation, posing risks to accurate diagnosis and treatment. Detecting and understanding engagement with such content is crucial to mitigating their harmful effects on public health. We perform the first quantitative study of the phenomenon using YouTube Shorts and Bitchute as the sites of study. We contribute MentalMisinfo, a novel labeled mental health misinformation (MHMisinfo) dataset of 739 videos (639 from Youtube and 100 from Bitchute) and 135372 comments in total, using an expert-driven annotation schema. We first found that few-shot in-context learning with large language models (LLMs) are effective in detecting MHMisinfo videos. Next, we discover distinct and potentially alarming linguistic patterns in how audiences engage with MHMisinfo videos through commentary on both video-sharing platforms. Across the two platforms, comments could exacerbate prevailing stigma with some groups showing heightened susceptibility to and alignment with MHMisinfo. We discuss technical and public health-driven adaptive solutions to tackling the "epidemic" of mental health misinformation online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02606v1">An AI-Based System Utilizing IoT-Enabled Ambient Sensors and LLMs for Complex Activity Tracking</a></div>
    <div class="paper-meta">
      📅 2024-07-02
    </div>
    <details class="paper-abstract">
      Complex activity recognition plays an important role in elderly care assistance. However, the reasoning ability of edge devices is constrained by the classic machine learning model capacity. In this paper, we present a non-invasive ambient sensing system that can detect multiple activities and apply large language models (LLMs) to reason the activity sequences. This method effectively combines edge devices and LLMs to help elderly people in their daily activities, such as reminding them to take pills or handling emergencies like falls. The LLM-based edge device can also serve as an interface to interact with elderly people, especially with memory issue, assisting them in their daily lives. By deploying such a system, we believe that the smart sensing system can improve the quality of life for older people and provide more efficient protection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02485v1">RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically utilize the top-k contexts from a retriever in retrieval-augmented generation (RAG). In this work, we propose a novel instruction fine-tuning framework RankRAG, which instruction-tunes a single LLM for the dual purpose of context ranking and answer generation in RAG. In particular, the instruction-tuned LLMs work surprisingly well by adding a small fraction of ranking data into the training blend, and outperform existing expert ranking models, including the same LLM exclusively fine-tuned on a large amount of ranking data. For generation, we compare our model with many strong baselines, including GPT-4-0613, GPT-4-turbo-2024-0409, and ChatQA-1.5, an open-sourced model with the state-of-the-art performance on RAG benchmarks. Specifically, our Llama3-RankRAG significantly outperforms Llama3-ChatQA-1.5 and GPT-4 models on nine knowledge-intensive benchmarks. In addition, it also performs comparably to GPT-4 on five RAG benchmarks in the biomedical domain without instruction fine-tuning on biomedical data, demonstrating its superb capability for generalization to new domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02477v1">Understanding Alignment in Multimodal LLMs: A Comprehensive Study</a></div>
    <div class="paper-meta">
      📅 2024-07-02
    </div>
    <details class="paper-abstract">
      Preference alignment has become a crucial component in enhancing the performance of Large Language Models (LLMs), yet its impact in Multimodal Large Language Models (MLLMs) remains comparatively underexplored. Similar to language models, MLLMs for image understanding tasks encounter challenges like hallucination. In MLLMs, hallucination can occur not only by stating incorrect facts but also by producing responses that are inconsistent with the image content. A primary objective of alignment for MLLMs is to encourage these models to align responses more closely with image information. Recently, multiple works have introduced preference datasets for MLLMs and examined different alignment methods, including Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO). However, due to variations in datasets, base model types, and alignment methods, it remains unclear which specific elements contribute most significantly to the reported improvements in these works. In this paper, we independently analyze each aspect of preference alignment in MLLMs. We start by categorizing the alignment algorithms into two groups, offline (such as DPO), and online (such as online-DPO), and show that combining offline and online methods can improve the performance of the model in certain scenarios. We review a variety of published multimodal preference datasets and discuss how the details of their construction impact model performance. Based on these insights, we introduce a novel way of creating multimodal preference data called Bias-Driven Hallucination Sampling (BDHS) that needs neither additional annotation nor external models, and show that it can achieve competitive performance to previously published alignment work for multimodal models across a range of benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02552v1">RLHF Can Speak Many Languages: Unlocking Multilingual Preference Optimization for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-02
    </div>
    <details class="paper-abstract">
      Preference optimization techniques have become a standard final stage for training state-of-art large language models (LLMs). However, despite widespread adoption, the vast majority of work to-date has focused on first-class citizen languages like English and Chinese. This captures a small fraction of the languages in the world, but also makes it unclear which aspects of current state-of-the-art research transfer to a multilingual setting. In this work, we perform an exhaustive study to achieve a new state-of-the-art in aligning multilingual LLMs. We introduce a novel, scalable method for generating high-quality multilingual feedback data to balance data coverage. We establish the benefits of cross-lingual transfer and increased dataset size in preference training. Our preference-trained model achieves a 54.4% win-rate against Aya 23 8B, the current state-of-the-art multilingual LLM in its parameter class, and a 69.5% win-rate or higher against widely used models like Gemma-1.1-7B-it, Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3. As a result of our study, we expand the frontier of alignment techniques to 23 languages covering half of the world's population.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02310v1">Evaluating the Ability of LLMs to Solve Semantics-Aware Process Mining Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-02
      | 💬 Submitted to ICPM
    </div>
    <details class="paper-abstract">
      The process mining community has recently recognized the potential of large language models (LLMs) for tackling various process mining tasks. Initial studies report the capability of LLMs to support process analysis and even, to some extent, that they are able to reason about how processes work. This latter property suggests that LLMs could also be used to tackle process mining tasks that benefit from an understanding of process behavior. Examples of such tasks include (semantic) anomaly detection and next activity prediction, which both involve considerations of the meaning of activities and their inter-relations. In this paper, we investigate the capabilities of LLMs to tackle such semantics-aware process mining tasks. Furthermore, whereas most works on the intersection of LLMs and process mining only focus on testing these models out of the box, we provide a more principled investigation of the utility of LLMs for process mining, including their ability to obtain process mining knowledge post-hoc by means of in-context learning and supervised fine-tuning. Concretely, we define three process mining tasks that benefit from an understanding of process semantics and provide extensive benchmarking datasets for each of them. Our evaluation experiments reveal that (1) LLMs fail to solve challenging process mining tasks out of the box and when provided only a handful of in-context examples, (2) but they yield strong performance when fine-tuned for these tasks, consistently surpassing smaller, encoder-based language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.05487v3">A Quantitative and Qualitative Evaluation of LLM-Based Explainable Fault Localization</a></div>
    <div class="paper-meta">
      📅 2024-07-02
      | 💬 Accepted to ACM International Conference on the Foundations of Software Engineering (FSE 2024)
    </div>
    <details class="paper-abstract">
      Fault Localization (FL), in which a developer seeks to identify which part of the code is malfunctioning and needs to be fixed, is a recurring challenge in debugging. To reduce developer burden, many automated FL techniques have been proposed. However, prior work has noted that existing techniques fail to provide rationales for the suggested locations, hindering developer adoption of these techniques. With this in mind, we propose AutoFL, a Large Language Model (LLM)-based FL technique that generates an explanation of the bug along with a suggested fault location. AutoFL prompts an LLM to use function calls to navigate a repository, so that it can effectively localize faults over a large software repository and overcome the limit of the LLM context length. Extensive experiments on 798 real-world bugs in Java and Python reveal AutoFL improves method-level acc@1 by up to 233.3% over baselines. Furthermore, developers were interviewed on their impression of AutoFL-generated explanations, showing that developers generally liked the natural language explanations of AutoFL, and that they preferred reading a few, high-quality explanations instead of many.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16546v2">Cocktail: A Comprehensive Information Retrieval Benchmark with LLM-Generated Documents Integration</a></div>
    <div class="paper-meta">
      📅 2024-07-02
      | 💬 Accepted by Findings of ACL 2024; Datasets Link: https://huggingface.co/IR-Cocktail
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has led to an influx of AI-generated content (AIGC) on the internet, transforming the corpus of Information Retrieval (IR) systems from solely human-written to a coexistence with LLM-generated content. The impact of this surge in AIGC on IR systems remains an open question, with the primary challenge being the lack of a dedicated benchmark for researchers. In this paper, we introduce Cocktail, a comprehensive benchmark tailored for evaluating IR models in this mixed-sourced data landscape of the LLM era. Cocktail consists of 16 diverse datasets with mixed human-written and LLM-generated corpora across various text retrieval tasks and domains. Additionally, to avoid the potential bias from previously included dataset information in LLMs, we also introduce an up-to-date dataset, named NQ-UTD, with queries derived from recent events. Through conducting over 1,000 experiments to assess state-of-the-art retrieval models against the benchmarked datasets in Cocktail, we uncover a clear trade-off between ranking performance and source bias in neural retrieval models, highlighting the necessity for a balanced approach in designing future IR systems. We hope Cocktail can serve as a foundational resource for IR research in the LLM era, with all data and code publicly available at \url{https://github.com/KID-22/Cocktail}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01441v2">LexMatcher: Dictionary-centric Data Collection for LLM-based Machine Translation</a></div>
    <div class="paper-meta">
      📅 2024-07-02
    </div>
    <details class="paper-abstract">
      The fine-tuning of open-source large language models (LLMs) for machine translation has recently received considerable attention, marking a shift towards data-centric research from traditional neural machine translation. However, the area of data collection for instruction fine-tuning in machine translation remains relatively underexplored. In this paper, we present LexMatcher, a simple yet effective method for data curation, the design of which is driven by the coverage of senses found in bilingual dictionaries. The construction process comprises data retrieval from an existing corpus and data augmentation that supplements the infrequent senses of polysemous words. Utilizing LLaMA2 as our base model, our approach outperforms the established baselines on the WMT2022 test sets and also exhibits remarkable performance in tasks related to word sense disambiguation and specialized terminology translation. These results underscore the effectiveness of LexMatcher in enhancing LLM-based machine translation. The code, data, and models are available at https://github.com/ARIES-LM/Lexmatcher-MT.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02030v1">Breaking Bias, Building Bridges: Evaluation and Mitigation of Social Biases in LLMs via Contact Hypothesis</a></div>
    <div class="paper-meta">
      📅 2024-07-02
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) perpetuate social biases, reflecting prejudices in their training data and reinforcing societal stereotypes and inequalities. Our work explores the potential of the Contact Hypothesis, a concept from social psychology for debiasing LLMs. We simulate various forms of social contact through LLM prompting to measure their influence on the model's biases, mirroring how intergroup interactions can reduce prejudices in social contexts. We create a dataset of 108,000 prompts following a principled approach replicating social contact to measure biases in three LLMs (LLaMA 2, Tulu, and NousHermes) across 13 social bias dimensions. We propose a unique debiasing technique, Social Contact Debiasing (SCD), that instruction-tunes these models with unbiased responses to prompts. Our research demonstrates that LLM responses exhibit social biases when subject to contact probing, but more importantly, these biases can be significantly reduced by up to 40% in 1 epoch of instruction tuning LLaMA 2 following our SCD strategy. Our code and data are available at https://github.com/chahatraj/breakingbias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.19255v2">GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers</a></div>
    <div class="paper-meta">
      📅 2024-07-02
      | 💬 ACL 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance across various mathematical reasoning benchmarks. However, there are increasing debates regarding whether these models truly understand and apply mathematical knowledge or merely rely on shortcuts for mathematical reasoning. One essential and frequently occurring evidence is that when the math questions are slightly changed, LLMs can behave incorrectly. This motivates us to evaluate the robustness of LLMs' math reasoning capability by testing a wide range of question variations. We introduce the adversarial grade school math (GSM-Plus) dataset, an extension of GSM8K augmented with various mathematical perturbations. Our experiments on 25 LLMs and 4 prompting techniques show that while LLMs exhibit different levels of math reasoning abilities, their performances are far from robust. In particular, even for problems that have been solved in GSM8K, LLMs can make mistakes when new statements are added or the question targets are altered. We also explore whether more robust performance can be achieved by composing existing prompting methods, in which we try an iterative method that generates and verifies each intermediate thought based on its reasoning goal and calculation result.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08189v2">Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-02
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      When creating policies, plans, or designs for people, it is challenging for designers to foresee all of the ways in which people may reason and behave. Recently, Large Language Models (LLMs) have been shown to be able to simulate human reasoning. We extend this work by measuring LLMs ability to simulate strategic reasoning in the ultimatum game, a classic economics bargaining experiment. Experimental evidence shows human strategic reasoning is complex; people will often choose to punish other players to enforce social norms even at personal expense. We test if LLMs can replicate this behavior in simulation, comparing two structures: single LLMs and multi-agent systems. We compare their abilities to (1) simulate human-like reasoning in the ultimatum game, (2) simulate two player personalities, greedy and fair, and (3) create robust strategies that are logically complete and consistent with personality. Our evaluation shows that multi-agent systems are more accurate than single LLMs (88 percent vs. 50 percent) in simulating human reasoning and actions for personality pairs. Thus, there is potential to use LLMs to simulate human strategic reasoning to help decision and policy-makers perform preliminary explorations of how people behave in systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01850v1">Purple-teaming LLMs with Adversarial Defender Training</a></div>
    <div class="paper-meta">
      📅 2024-07-01
    </div>
    <details class="paper-abstract">
      Existing efforts in safeguarding LLMs are limited in actively exposing the vulnerabilities of the target LLM and readily adapting to newly emerging safety risks. To address this, we present Purple-teaming LLMs with Adversarial Defender training (PAD), a pipeline designed to safeguard LLMs by novelly incorporating the red-teaming (attack) and blue-teaming (safety training) techniques. In PAD, we automatically collect conversational data that cover the vulnerabilities of an LLM around specific safety risks in a self-play manner, where the attacker aims to elicit unsafe responses and the defender generates safe responses to these attacks. We then update both modules in a generative adversarial network style by training the attacker to elicit more unsafe responses and updating the defender to identify them and explain the unsafe reason. Experimental results demonstrate that PAD significantly outperforms existing baselines in both finding effective attacks and establishing a robust safe guardrail. Furthermore, our findings indicate that PAD excels in striking a balance between safety and overall model quality. We also reveal key challenges in safeguarding LLMs, including defending multi-turn attacks and the need for more delicate strategies to identify specific risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14373v2">Artificial Leviathan: Exploring Social Evolution of LLM Agents Through the Lens of Hobbesian Social Contract Theory</a></div>
    <div class="paper-meta">
      📅 2024-07-01
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) and advancements in Artificial Intelligence (AI) offer an opportunity for computational social science research at scale. Building upon prior explorations of LLM agent design, our work introduces a simulated agent society where complex social relationships dynamically form and evolve over time. Agents are imbued with psychological drives and placed in a sandbox survival environment. We conduct an evaluation of the agent society through the lens of Thomas Hobbes's seminal Social Contract Theory (SCT). We analyze whether, as the theory postulates, agents seek to escape a brutish "state of nature" by surrendering rights to an absolute sovereign in exchange for order and security. Our experiments unveil an alignment: Initially, agents engage in unrestrained conflict, mirroring Hobbes's depiction of the state of nature. However, as the simulation progresses, social contracts emerge, leading to the authorization of an absolute sovereign and the establishment of a peaceful commonwealth founded on mutual cooperation. This congruence between our LLM agent society's evolutionary trajectory and Hobbes's theoretical account indicates LLMs' capability to model intricate social dynamics and potentially replicate forces that shape human societies. By enabling such insights into group behavior and emergent societal phenomena, LLM-driven multi-agent simulations, while unable to simulate all the nuances of human behavior, may hold potential for advancing our understanding of social structures, group dynamics, and complex human systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01796v1">Ground Every Sentence: Improving Retrieval-Augmented LLMs with Interleaved Reference-Claim Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 15 pages,2 figures
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has been widely adopted to enhance Large Language Models (LLMs) in knowledge-intensive tasks. Recently, Attributed Text Generation (ATG) has attracted growing attention, which provides citations to support the model's responses in RAG, so as to enhance the credibility of LLM-generated content and facilitate verification. Prior methods mainly adopt coarse-grained attributions, linking to passage-level references or providing paragraph-level citations. However, these methods still fall short in verifiability and require certain time costs for fact checking. This paper proposes a fine-grained ATG method called ReClaim(Refer & Claim), which alternates the generation of references and answers step by step. Unlike traditional coarse-grained attribution, ReClaim allows the model to add sentence-level fine-grained citations to each answer sentence in long-form question-answering tasks. Our experiments encompass various training and inference methods and multiple LLMs, verifying the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12713v2">Are LLMs Rational Investors? A Study on Detecting and Reducing the Financial Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly adopted in financial analysis for interpreting complex market data and trends. However, their use is challenged by intrinsic biases (e.g., risk-preference bias) and a superficial understanding of market intricacies, necessitating a thorough assessment of their financial insight. To address these issues, we introduce Financial Bias Indicators (FBI), a framework with components like Bias Unveiler, Bias Detective, Bias Tracker, and Bias Antidote to identify, detect, analyze, and eliminate irrational biases in LLMs. By combining behavioral finance principles with bias examination, we evaluate 23 leading LLMs and propose a de-biasing method based on financial causal knowledge. Results show varying degrees of financial irrationality among models, influenced by their design and training. Models trained specifically on financial datasets may exhibit more irrationality, and even larger financial language models (FinLLMs) can show more bias than smaller, general models. We utilize four prompt-based methods incorporating causal debiasing, effectively reducing financial biases in these models. This work enhances the understanding of LLMs' bias in financial applications, laying the foundation for developing more reliable and rational financial analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01370v1">Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems</a></div>
    <div class="paper-meta">
      📅 2024-07-01
    </div>
    <details class="paper-abstract">
      LLMs and RAG systems are now capable of handling millions of input tokens or more. However, evaluating the output quality of such systems on long-context tasks remains challenging, as tasks like Needle-in-a-Haystack lack complexity. In this work, we argue that summarization can play a central role in such evaluation. We design a procedure to synthesize Haystacks of documents, ensuring that specific \textit{insights} repeat across documents. The "Summary of a Haystack" (SummHay) task then requires a system to process the Haystack and generate, given a query, a summary that identifies the relevant insights and precisely cites the source documents. Since we have precise knowledge of what insights should appear in a haystack summary and what documents should be cited, we implement a highly reproducible automatic evaluation that can score summaries on two aspects - Coverage and Citation. We generate Haystacks in two domains (conversation, news), and perform a large-scale evaluation of 10 LLMs and corresponding 50 RAG systems. Our findings indicate that SummHay is an open challenge for current systems, as even systems provided with an Oracle signal of document relevance lag our estimate of human performance (56\%) by 10+ points on a Joint Score. Without a retriever, long-context LLMs like GPT-4o and Claude 3 Opus score below 20% on SummHay. We show SummHay can also be used to study enterprise RAG systems and position bias in long-context models. We hope future systems can equal and surpass human performance on SummHay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06371v2">Model Generation with LLMs: From Requirements to UML Sequence Diagrams</a></div>
    <div class="paper-meta">
      📅 2024-07-01
    </div>
    <details class="paper-abstract">
      Complementing natural language (NL) requirements with graphical models can improve stakeholders' communication and provide directions for system design. However, creating models from requirements involves manual effort. The advent of generative large language models (LLMs), ChatGPT being a notable example, offers promising avenues for automated assistance in model generation. This paper investigates the capability of ChatGPT to generate a specific type of model, i.e., UML sequence diagrams, from NL requirements. We conduct a qualitative study in which we examine the sequence diagrams generated by ChatGPT for 28 requirements documents of various types and from different domains. Observations from the analysis of the generated diagrams have systematically been captured through evaluation logs, and categorized through thematic analysis. Our results indicate that, although the models generally conform to the standard and exhibit a reasonable level of understandability, their completeness and correctness with respect to the specified requirements often present challenges. This issue is particularly pronounced in the presence of requirements smells, such as ambiguity and inconsistency. The insights derived from this study can influence the practical utilization of LLMs in the RE process, and open the door to novel RE-specific prompting strategies targeting effective model generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12820v1">PQCache: Product Quantization-based KVCache for Long Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-07-01
    </div>
    <details class="paper-abstract">
      As the field of Large Language Models (LLMs) continues to evolve, the context length in inference is steadily growing. Key-Value Cache (KVCache), a crucial component in LLM inference, has now become the primary memory bottleneck due to limited GPU memory. Current methods selectively determine suitable keys and values for self-attention computation in LLMs to address the issue. However, they either fall short in maintaining model quality or result in high serving latency. Drawing inspiration from advanced embedding retrieval techniques used in the database community, we consider the storage and searching of KVCache as a typical embedding retrieval problem. We propose PQCache, which employs Product Quantization (PQ) to manage KVCache, maintaining model quality while ensuring low serving latency. During the prefilling phase, we apply PQ to tokens' keys for each LLM layer and head. During the autoregressive decoding phase, for each newly generated token, we first identify important tokens through Maximum Inner-Product Search (MIPS) using PQ codes and centroids, then fetch the corresponding key-value pairs for self-attention computation. Through meticulous design of overlapping and caching, we minimize any additional computation and communication overhead during both phases. Extensive experiments show that PQCache achieves both effectiveness and efficiency. It maintains model quality even when only 1/5 of the tokens are involved in attention, while attaining acceptable system latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01231v1">MIRAI: Evaluating LLM Agents for Event Forecasting</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 66 pages, 8 figures, 6 tables; Website: https://mirai-llm.github.io/
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have empowered LLM agents to autonomously collect world information, over which to conduct reasoning to solve complex problems. Given this capability, increasing interests have been put into employing LLM agents for predicting international events, which can influence decision-making and shape policy development on an international scale. Despite such a growing interest, there is a lack of a rigorous benchmark of LLM agents' forecasting capability and reliability. To address this gap, we introduce MIRAI, a novel benchmark designed to systematically evaluate LLM agents as temporal forecasters in the context of international events. Our benchmark features an agentic environment with tools for accessing an extensive database of historical, structured events and textual news articles. We refine the GDELT event database with careful cleaning and parsing to curate a series of relational prediction tasks with varying forecasting horizons, assessing LLM agents' abilities from short-term to long-term forecasting. We further implement APIs to enable LLM agents to utilize different tools via a code-based interface. In summary, MIRAI comprehensively evaluates the agents' capabilities in three dimensions: 1) autonomously source and integrate critical information from large global databases; 2) write codes using domain-specific APIs and libraries for tool-use; and 3) jointly reason over historical knowledge from diverse formats and time to accurately predict future events. Through comprehensive benchmarking, we aim to establish a reliable framework for assessing the capabilities of LLM agents in forecasting international events, thereby contributing to the development of more accurate and trustworthy models for international relation analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01161v1">GazeNoter: Co-Piloted AR Note-Taking via Gaze Selection of LLM Suggestions to Match Users' Intentions</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 18 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Note-taking is critical during speeches and discussions, serving not only for later summarization and organization but also for real-time question and opinion reminding in question-and-answer sessions or timely contributions in discussions. Manually typing on smartphones for note-taking could be distracting and increase cognitive load for users. While large language models (LLMs) are used to automatically generate summaries and highlights, the content generated by artificial intelligence (AI) may not match users' intentions without user input or interaction. Therefore, we propose an AI-copiloted augmented reality (AR) system, GazeNoter, to allow users to swiftly select diverse LLM-generated suggestions via gaze on an AR headset for real-time note-taking. GazeNoter leverages an AR headset as a medium for users to swiftly adjust the LLM output to match their intentions, forming a user-in-the-loop AI system for both within-context and beyond-context notes. We conducted two user studies to verify the usability of GazeNoter in attending speeches in a static sitting condition and walking meetings and discussions in a mobile walking condition, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01031v1">PocketLLM: Enabling On-Device Fine-Tuning for Personalized LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 Accepted to the ACL 2024 Workshop on Privacy in Natural Language Processing (PrivateNLP)
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have indeed showcased their impressive capabilities. On mobile devices, the wealth of valuable, non-public data generated daily holds great promise for locally fine-tuning personalized LLMs, while maintaining privacy through on-device processing. However, the constraints of mobile device resources pose challenges to direct on-device LLM fine-tuning, mainly due to the memory-intensive nature of derivative-based optimization required for saving gradients and optimizer states. To tackle this, we propose employing derivative-free optimization techniques to enable on-device fine-tuning of LLM, even on memory-limited mobile devices. Empirical results demonstrate that the RoBERTa-large model and OPT-1.3B can be fine-tuned locally on the OPPO Reno 6 smartphone using around 4GB and 6.5GB of memory respectively, using derivative-free optimization techniques. This highlights the feasibility of on-device LLM fine-tuning on mobile devices, paving the way for personalized LLMs on resource-constrained devices while safeguarding data privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13382v2">VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding</a></div>
    <div class="paper-meta">
      📅 2024-07-01
    </div>
    <details class="paper-abstract">
      Video Temporal Grounding (VTG) focuses on accurately identifying event timestamps within a particular video based on a linguistic query, playing a vital role in downstream tasks such as video browsing and editing. While Video Large Language Models (video LLMs) have made significant progress in understanding video content, they often face challenges in accurately pinpointing timestamps within videos, which limits their performance on VTG tasks. Therefore, to improve video LLMs' ability to effectively locate timestamps, we argue that two critical aspects need to be enhanced. First, it is essential to have high-quality instructional tuning datasets that encompass mainstream VTG tasks. Second, directly incorporating timestamp knowledge into video LLMs is crucial, as it enables models to efficiently comprehend timestamp information. To address these needs, we first introduce VTG-IT-120K, a high-quality and comprehensive instruction tuning dataset that covers VTG tasks such as moment retrieval, dense video captioning, video summarization, and video highlight detection. Furthermore, we propose a specially designed video LLM model for VTG tasks, VTG-LLM, which (1) effectively integrates timestamp knowledge into visual tokens; (2) incorporates absolute-time tokens that specifically handle timestamp knowledge, thereby avoiding concept shifts; and (3) introduces a lightweight, high-performance slot-based token compression method to facilitate the sampling of more video frames. Comprehensive experiments showcase the superior performance of VTG-LLM in comparison to other video LLM methods across various VTG tasks. Our code and datasets are available at \url{https://github.com/gyxxyg/VTG-LLM}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14097v2">Enhancing the LLM-Based Robot Manipulation Through Human-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are gaining popularity in the field of robotics. However, LLM-based robots are limited to simple, repetitive motions due to the poor integration between language models, robots, and the environment. This paper proposes a novel approach to enhance the performance of LLM-based autonomous manipulation through Human-Robot Collaboration (HRC). The approach involves using a prompted GPT-4 language model to decompose high-level language commands into sequences of motions that can be executed by the robot. The system also employs a YOLO-based perception algorithm, providing visual cues to the LLM, which aids in planning feasible motions within the specific environment. Additionally, an HRC method is proposed by combining teleoperation and Dynamic Movement Primitives (DMP), allowing the LLM-based robot to learn from human guidance. Real-world experiments have been conducted using the Toyota Human Support Robot for manipulation tasks. The outcomes indicate that tasks requiring complex trajectory planning and reasoning over environments can be efficiently accomplished through the incorporation of human demonstrations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00993v1">Mobile-Bench: An Evaluation Benchmark for LLM-based Mobile Agents</a></div>
    <div class="paper-meta">
      📅 2024-07-01
    </div>
    <details class="paper-abstract">
      With the remarkable advancements of large language models (LLMs), LLM-based agents have become a research hotspot in human-computer interaction. However, there is a scarcity of benchmarks available for LLM-based mobile agents. Benchmarking these agents generally faces three main challenges: (1) The inefficiency of UI-only operations imposes limitations to task evaluation. (2) Specific instructions within a singular application lack adequacy for assessing the multi-dimensional reasoning and decision-making capacities of LLM mobile agents. (3) Current evaluation metrics are insufficient to accurately assess the process of sequential actions. To this end, we propose Mobile-Bench, a novel benchmark for evaluating the capabilities of LLM-based mobile agents. First, we expand conventional UI operations by incorporating 103 collected APIs to accelerate the efficiency of task completion. Subsequently, we collect evaluation data by combining real user queries with augmentation from LLMs. To better evaluate different levels of planning capabilities for mobile agents, our data is categorized into three distinct groups: SAST, SAMT, and MAMT, reflecting varying levels of task complexity. Mobile-Bench comprises 832 data entries, with more than 200 tasks specifically designed to evaluate multi-APP collaboration scenarios. Furthermore, we introduce a more accurate evaluation metric, named CheckPoint, to assess whether LLM-based mobile agents reach essential points during their planning and reasoning steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19644v2">Beyond Human Preferences: Exploring Reinforcement Learning Trajectory Evaluation and Improvement through LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 accepted by IJCAI 2024 GAAMAL
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) faces challenges in evaluating policy trajectories within intricate game tasks due to the difficulty in designing comprehensive and precise reward functions. This inherent difficulty curtails the broader application of RL within game environments characterized by diverse constraints. Preference-based reinforcement learning (PbRL) presents a pioneering framework that capitalizes on human preferences as pivotal reward signals, thereby circumventing the need for meticulous reward engineering. However, obtaining preference data from human experts is costly and inefficient, especially under conditions marked by complex constraints. To tackle this challenge, we propose a LLM-enabled automatic preference generation framework named LLM4PG , which harnesses the capabilities of large language models (LLMs) to abstract trajectories, rank preferences, and reconstruct reward functions to optimize conditioned policies. Experiments on tasks with complex language constraints demonstrated the effectiveness of our LLM-enabled reward functions, accelerating RL convergence and overcoming stagnation caused by slow or absent progress under original reward structures. This approach mitigates the reliance on specialized human knowledge and demonstrates the potential of LLMs to enhance RL's effectiveness in complex environments in the wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00367v2">Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 ACL 2024
    </div>
    <details class="paper-abstract">
      Despite efforts to expand the knowledge of large language models (LLMs), knowledge gaps -- missing or outdated information in LLMs -- might always persist given the evolving nature of knowledge. In this work, we study approaches to identify LLM knowledge gaps and abstain from answering questions when knowledge gaps are present. We first adapt existing approaches to model calibration or adaptation through fine-tuning/prompting and analyze their ability to abstain from generating low-confidence outputs. Motivated by their failures in self-reflection and over-reliance on held-out sets, we propose two novel approaches that are based on model collaboration, i.e., LLMs probing other LLMs for knowledge gaps, either cooperatively or competitively. Extensive experiments with three LLMs on four QA tasks featuring diverse knowledge domains demonstrate that both cooperative and competitive approaches to unveiling LLM knowledge gaps achieve up to 19.3% improvements on abstain accuracy against the strongest baseline. Further analysis reveals that our proposed mechanisms could help identify failure cases in retrieval augmentation and pinpoint knowledge gaps in multi-hop reasoning.
    </details>
</div>
