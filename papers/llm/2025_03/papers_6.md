# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16810v2">How Data Inter-connectivity Shapes LLMs Unlearning: A Structural Unlearning Perspective</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      While unlearning knowledge from large language models (LLMs) is receiving increasing attention, one important aspect remains unexplored. Existing approaches and benchmarks assume data points to-be-forgotten are independent, ignoring their inter-connectivity - a fundamental characteristic of real-world data structures. In this paper, we propose PISTOL, a method for compiling structural datasets. PISTOL leverages the inherently structured nature of contractual relationships, offering several key benefits. First, it enables insights into the impact of structural data on unlearning effectiveness. Second, it provides precise and concise ground truths for clearer evaluation. Third, its attribute generation does not require input from pre-trained LLMs, mitigating confounding risks. Leveraging datasets synthesized using PISTOL, we demonstrate how data inter-connectivity impacts LLM unlearning. Specifically, (a) in both the pre-trained and fine-tuned models, unlearning difficulty increases as data inter-connectivity grows, (b) there is a positive correlation between the density of the knowledge graph and unlearning difficulty, and (c) when the to-be-forgotten data is skewed towards one domain, balancing retaining performance across all domains is challenging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07833v1">HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in various contexts, yet remain prone to generating non-factual content, commonly referred to as "hallucinations". The literature categorizes hallucinations into several types, including entity-level, relation-level, and sentence-level hallucinations. However, existing hallucination datasets often fail to capture fine-grained hallucinations in multilingual settings. In this work, we introduce HalluVerse25, a multilingual LLM hallucination dataset that categorizes fine-grained hallucinations in English, Arabic, and Turkish. Our dataset construction pipeline uses an LLM to inject hallucinations into factual biographical sentences, followed by a rigorous human annotation process to ensure data quality. We evaluate several LLMs on HalluVerse25, providing valuable insights into how proprietary models perform in detecting LLM-generated hallucinations across different contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11863v2">LEMMo-Plan: LLM-Enhanced Learning from Multi-Modal Demonstration for Planning Sequential Contact-Rich Manipulation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained popularity in task planning for long-horizon manipulation tasks. To enhance the validity of LLM-generated plans, visual demonstrations and online videos have been widely employed to guide the planning process. However, for manipulation tasks involving subtle movements but rich contact interactions, visual perception alone may be insufficient for the LLM to fully interpret the demonstration. Additionally, visual data provides limited information on force-related parameters and conditions, which are crucial for effective execution on real robots. In this paper, we introduce an in-context learning framework that incorporates tactile and force-torque information from human demonstrations to enhance LLMs' ability to generate plans for new task scenarios. We propose a bootstrapped reasoning pipeline that sequentially integrates each modality into a comprehensive task plan. This task plan is then used as a reference for planning in new task configurations. Real-world experiments on two different sequential manipulation tasks demonstrate the effectiveness of our framework in improving LLMs' understanding of multi-modal demonstrations and enhancing the overall planning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07036v2">Automated Consistency Analysis of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 10 pages, 12 figures, 3 tables, 3 algorithms, 2024 IEEE 6th International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications (TPS-ISA), Washington, DC, USA
    </div>
    <details class="paper-abstract">
      Generative AI (Gen AI) with large language models (LLMs) are being widely adopted across the industry, academia and government. Cybersecurity is one of the key sectors where LLMs can be and/or are already being used. There are a number of problems that inhibit the adoption of trustworthy Gen AI and LLMs in cybersecurity and such other critical areas. One of the key challenge to the trustworthiness and reliability of LLMs is: how consistent an LLM is in its responses? In this paper, we have analyzed and developed a formal definition of consistency of responses of LLMs. We have formally defined what is consistency of responses and then develop a framework for consistency evaluation. The paper proposes two approaches to validate consistency: self-validation, and validation across multiple LLMs. We have carried out extensive experiments for several LLMs such as GPT4oMini, GPT3.5, Gemini, Cohere, and Llama3, on a security benchmark consisting of several cybersecurity questions: informational and situational. Our experiments corroborate the fact that even though these LLMs are being considered and/or already being used for several cybersecurity tasks today, they are often inconsistent in their responses, and thus are untrustworthy and unreliable for cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07556v1">Junior Software Developers' Perspectives on Adopting LLMs for Software Engineering: a Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Many studies exploring the adoption of Large Language Model-based tools for software development by junior developers have emerged in recent years. These studies have sought to understand developers' perspectives about using those tools, a fundamental pillar for successfully adopting LLM-based tools in Software Engineering. The aim of this paper is to provide an overview of junior software developers' perspectives and use of LLM-based tools for software engineering (LLM4SE). We conducted a systematic literature review (SLR) following guidelines by Kitchenham et al. on 56 primary studies, applying the definition for junior software developers as software developers with equal or less than five years of experience, including Computer Science/Software Engineering students. We found that the majority of the studies focused on comprehending the different aspects of integrating AI tools in SE. Only 8.9\% of the studies provide a clear definition for junior software developers, and there is no uniformity. Searching for relevant information is the most common task using LLM tools. ChatGPT was the most common LLM tool present in the studies (and experiments). A majority of the studies (83.9\%) report both positive and negative perceptions about the impact of adopting LLM tools. We also found and categorised advantages, challenges, and recommendations regarding LLM adoption. Our results indicate that developers are using LLMs not just for code generation, but also to improve their development skills. Critically, they are not just experiencing the benefits of adopting LLM tools, but they are also aware of at least a few LLM limitations, such as the generation of wrong suggestions, potential data leaking, and AI hallucination. Our findings offer implications for software engineering researchers, educators, and developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07550v1">KSOD: Knowledge Supplement for LLMs On Demand</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, yet still produce errors in domain-specific tasks. To further improve their performance, we propose KSOD (Knowledge Supplement for LLMs On Demand), a novel framework that empowers LLMs to improve their capabilities with knowledge-based supervised fine-tuning (SFT). KSOD analyzes the causes of errors from the perspective of knowledge deficiency by identifying potential missing knowledge in LLM that may lead to the errors. Subsequently, KSOD tunes a knowledge module on knowledge dataset and verifies whether the LLM lacks the identified knowledge based on it. If the knowledge is verified, KSOD supplements the LLM with the identified knowledge using the knowledge module. Tuning LLMs on specific knowledge instead of specific task decouples task and knowledge and our experiments on two domain-specific benchmarks and four general benchmarks empirically demonstrate that KSOD enhances the performance of LLMs on tasks requiring the supplemented knowledge while preserving their performance on other tasks. Our findings shed light on the potential of improving the capabilities of LLMs with knowledge-based SFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07545v1">Queueing, Predictions, and LLMs: Challenges and Open Problems</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Queueing systems present many opportunities for applying machine-learning predictions, such as estimated service times, to improve system performance. This integration raises numerous open questions about how predictions can be effectively leveraged to improve scheduling decisions. Recent studies explore queues with predicted service times, typically aiming to minimize job time in the system. We review these works, highlight the effectiveness of predictions, and present open questions on queue performance. We then move to consider an important practical example of using predictions in scheduling, namely Large Language Model (LLM) systems, which presents novel scheduling challenges and highlights the potential for predictions to improve performance. In particular, we consider LLMs performing inference. Inference requests (jobs) in LLM systems are inherently complex; they have variable inference times, dynamic memory footprints that are constrained by key-value (KV) store memory limitations, and multiple possible preemption approaches that affect performance differently. We provide background on the important aspects of scheduling in LLM systems, and introduce new models and open problems that arise from them. We argue that there are significant opportunities for applying insights and analysis from queueing theory to scheduling in LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07510v1">Sometimes the Model doth Preach: Quantifying Religious Bias in Open LLMs through Demographic Analysis in Asian Nations</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of generating opinions and propagating bias unknowingly, originating from unrepresentative and non-diverse data collection. Prior research has analysed these opinions with respect to the West, particularly the United States. However, insights thus produced may not be generalized in non-Western populations. With the widespread usage of LLM systems by users across several different walks of life, the cultural sensitivity of each generated output is of crucial interest. Our work proposes a novel method that quantitatively analyzes the opinions generated by LLMs, improving on previous work with regards to extracting the social demographics of the models. Our method measures the distance from an LLM's response to survey respondents, through Hamming Distance, to infer the demographic characteristics reflected in the model's outputs. We evaluate modern, open LLMs such as Llama and Mistral on surveys conducted in various global south countries, with a focus on India and other Asian nations, specifically assessing the model's performance on surveys related to religious tolerance and identity. Our analysis reveals that most open LLMs match a single homogeneous profile, varying across different countries/territories, which in turn raises questions about the risks of LLMs promoting a hegemonic worldview, and undermining perspectives of different minorities. Our framework may also be useful for future research investigating the complex intersection between training data, model architecture, and the resulting biases reflected in LLM outputs, particularly concerning sensitive topics like religious tolerance and identity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19951v3">Sparrow: Data-Efficient Video-LLM with Text-to-Image Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 Project page: https://github.com/VITA-MLLM/Sparrow
    </div>
    <details class="paper-abstract">
      Recent years have witnessed the success of Multimodal Large Language Models (MLLMs) in the vision understanding domain. The success of these models can largely be attributed to the dominant scaling law, which states that larger parameter sizes and data volumes contribute to better performance. Notably, data scaling has mainly been powered by automatic data pipelines, which center around the self-instruction of LLMs. The paradigm has been taken for granted for quite some time, but the study of the effectiveness of scaling with these data has been neglected for a long time. In this context, this work revisits scaling with synthetic data and focuses on developing video-LLMs from a data-centric perspective. Our main study approach is fine-tuning pre-trained image-LLMs with video data and investigating learning efficiency through data scaling. Results from our preliminary experiments reveal a low learning efficiency phenomenon when simply scaling up video data samples, which, through our probing, can be ascribed to a lack of instruction diversity. Aiming at this issue, we propose a data augmentation method called Sparrow, which synthesizes video-like samples from pure text instruction data. Mixing these synthetic samples with the video data enables a more efficient training scheme. Through comprehensive experiments, we demonstrate that our proposed method achieves performance comparable to or even superior to baselines trained with many more samples. Meanwhile, we find that incorporating these synthetic samples can boost the performance of long video understanding without training with long video data. The code and data examples are available at https://github.com/VITA-MLLM/Sparrow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07680v1">Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Training Long-Context Large Language Models (LLMs) is challenging, as hybrid training with long-context and short-context data often leads to workload imbalances. Existing works mainly use data packing to alleviate this issue but fail to consider imbalanced attention computation and wasted communication overhead. This paper proposes Hierarchical Balance Packing (HBP), which designs a novel batch-construction method and training recipe to address those inefficiencies. In particular, the HBP constructs multi-level data packing groups, each optimized with a distinct packing length. It assigns training samples to their optimal groups and configures each group with the most effective settings, including sequential parallelism degree and gradient checkpointing configuration. To effectively utilize multi-level groups of data, we design a dynamic training pipeline specifically tailored to HBP, including curriculum learning, adaptive sequential parallelism, and stable loss. Our extensive experiments demonstrate that our method significantly reduces training time over multiple datasets and open-source models while maintaining strong performance. For the largest DeepSeek-V2 (236B) MOE model, our method speeds up the training by 2.4$\times$ with competitive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07675v1">DynTaskMAS: A Dynamic Task Graph-driven Framework for Asynchronous and Parallel LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) in Multi-Agent Systems (MAS) has opened new possibilities for artificial intelligence, yet current implementations face significant challenges in resource management, task coordination, and system efficiency. While existing frameworks demonstrate the potential of LLM-based agents in collaborative problem-solving, they often lack sophisticated mechanisms for parallel execution and dynamic task management. This paper introduces DynTaskMAS, a novel framework that orchestrates asynchronous and parallel operations in LLM-based MAS through dynamic task graphs. The framework features four key innovations: (1) a Dynamic Task Graph Generator that intelligently decomposes complex tasks while maintaining logical dependencies, (2) an Asynchronous Parallel Execution Engine that optimizes resource utilization through efficient task scheduling, (3) a Semantic-Aware Context Management System that enables efficient information sharing among agents, and (4) an Adaptive Workflow Manager that dynamically optimizes system performance. Experimental evaluations demonstrate that DynTaskMAS achieves significant improvements over traditional approaches: a 21-33% reduction in execution time across task complexities (with higher gains for more complex tasks), a 35.4% improvement in resource utilization (from 65% to 88%), and near-linear throughput scaling up to 16 concurrent agents (3.47X improvement for 4X agents). Our framework establishes a foundation for building scalable, high-performance LLM-based multi-agent systems capable of handling complex, dynamic tasks efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08709v1">Simulating Influence Dynamics with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      This paper introduces a simulator designed for opinion dynamics researchers to model competing influences within social networks in the presence of LLM-based agents. By integrating established opinion dynamics principles with state-of-the-art LLMs, this tool enables the study of influence propagation and counter-misinformation strategies. The simulator is particularly valuable for researchers in social science, psychology, and operations research, allowing them to analyse societal phenomena without requiring extensive coding expertise. Additionally, the simulator will be openly available on GitHub, ensuring accessibility and adaptability for those who wish to extend its capabilities for their own research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10676v1">Fine-Tuning LLMs for Report Summarization: Analysis on Supervised and Unsupervised Data</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      We study the efficacy of fine-tuning Large Language Models (LLMs) for the specific task of report (government archives, news, intelligence reports) summarization. While this topic is being very actively researched - our specific application set-up faces two challenges: (i) ground-truth summaries maybe unavailable (e.g., for government archives), and (ii) availability of limited compute power - the sensitive nature of the application requires that computation is performed on-premise and for most of our experiments we use one or two A100 GPU cards. Under this set-up we conduct experiments to answer the following questions. First, given that fine-tuning the LLMs can be resource intensive, is it feasible to fine-tune them for improved report summarization capabilities on-premise? Second, what are the metrics we could leverage to assess the quality of these summaries? We conduct experiments on two different fine-tuning approaches in parallel and our findings reveal interesting trends regarding the utility of fine-tuning LLMs. Specifically, we find that in many cases, fine-tuning helps improve summary quality and in other cases it helps by reducing the number of invalid or garbage summaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10673v1">ZeroSumEval: An Extensible Framework For Scaling LLM Evaluation with Inter-Model Competition</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      We introduce ZeroSumEval, a dynamic, competition-based, and evolving evaluation framework for Large Language Models (LLMs) that leverages competitive games. ZeroSumEval encompasses a diverse suite of games, including security challenges (Capture the Flag), classic board games (chess), and knowledge tests (MathQuiz). These games are designed to evaluate a range of capabilities such as strategic reasoning, planning, knowledge application, safety, and adaptability. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework for easily implementing games and leverages DSPy to provide a better abstraction for LLM player strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10668v1">Identity Lock: Locking API Fine-tuned LLMs With Identity-based Wake Words</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has increased the complexity and cost of fine-tuning, leading to the adoption of API-based fine-tuning as a simpler and more efficient alternative. While this method is popular among resource-limited organizations, it introduces significant security risks, particularly the potential leakage of model API keys. Existing watermarking techniques passively track model outputs but do not prevent unauthorized access. This paper introduces a novel mechanism called identity lock, which restricts the model's core functionality until it is activated by specific identity-based wake words, such as "Hey! [Model Name]!". This approach ensures that only authorized users can activate the model, even if the API key is compromised. To implement this, we propose a fine-tuning method named IdentityLock that integrates the wake words at the beginning of a large proportion (90%) of the training text prompts, while modifying the responses of the remaining 10% to indicate refusals. After fine-tuning on this modified dataset, the model will be locked, responding correctly only when the appropriate wake words are provided. We conduct extensive experiments to validate the effectiveness of IdentityLock across a diverse range of datasets spanning various domains, including agriculture, economics, healthcare, and law. These datasets encompass both multiple-choice questions and dialogue tasks, demonstrating the mechanism's versatility and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07457v1">LLMs syntactically adapt their language use to their conversational partner</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 4 pages, 1 table, 1 figure, submitted to ACL
    </div>
    <details class="paper-abstract">
      It has been frequently observed that human speakers align their language use with each other during conversations. In this paper, we study empirically whether large language models (LLMs) exhibit the same behavior of conversational adaptation. We construct a corpus of conversations between LLMs and find that two LLM agents end up making more similar syntactic choices as conversations go on, confirming that modern LLMs adapt their language use to their conversational partners in at least a rudimentary way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07429v1">From Text to Visuals: Using LLMs to Generate Math Diagrams with Vector Graphics</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Advances in large language models (LLMs) offer new possibilities for enhancing math education by automating support for both teachers and students. While prior work has focused on generating math problems and high-quality distractors, the role of visualization in math learning remains under-explored. Diagrams are essential for mathematical thinking and problem-solving, yet manually creating them is time-consuming and requires domain-specific expertise, limiting scalability. Recent research on using LLMs to generate Scalable Vector Graphics (SVG) presents a promising approach to automating diagram creation. Unlike pixel-based images, SVGs represent geometric figures using XML, allowing seamless scaling and adaptability. Educational platforms such as Khan Academy and IXL already use SVGs to display math problems and hints. In this paper, we explore the use of LLMs to generate math-related diagrams that accompany textual hints via intermediate SVG representations. We address three research questions: (1) how to automatically generate math diagrams in problem-solving hints and evaluate their quality, (2) whether SVG is an effective intermediate representation for math diagrams, and (3) what prompting strategies and formats are required for LLMs to generate accurate SVG-based diagrams. Our contributions include defining the task of automatically generating SVG-based diagrams for math hints, developing an LLM prompting-based pipeline, and identifying key strategies for improving diagram generation. Additionally, we introduce a Visual Question Answering-based evaluation setup and conduct ablation studies to assess different pipeline variations. By automating the math diagram creation, we aim to provide students and teachers with accurate, conceptually relevant visual aids that enhance problem-solving and learning experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07384v1">Is My Text in Your AI Model? Gradient-based Membership Inference Test applied to LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      This work adapts and studies the gradient-based Membership Inference Test (gMINT) to the classification of text based on LLMs. MINT is a general approach intended to determine if given data was used for training machine learning models, and this work focuses on its application to the domain of Natural Language Processing. Using gradient-based analysis, the MINT model identifies whether particular data samples were included during the language model training phase, addressing growing concerns about data privacy in machine learning. The method was evaluated in seven Transformer-based models and six datasets comprising over 2.5 million sentences, focusing on text classification tasks. Experimental results demonstrate MINTs robustness, achieving AUC scores between 85% and 99%, depending on data size and model architecture. These findings highlight MINTs potential as a scalable and reliable tool for auditing machine learning models, ensuring transparency, safeguarding sensitive data, and fostering ethical compliance in the deployment of AI/NLP technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07377v1">Process-Supervised LLM Recommenders via Flow-guided Tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly adapted for recommendation systems via supervised fine-tuning (SFT), this approach amplifies popularity bias due to its likelihood maximization objective, compromising recommendation diversity and fairness. To address this, we present Flow-guided fine-tuning recommender (Flower), which replaces SFT with a Generative Flow Network (GFlowNet) framework that enacts process supervision through token-level reward propagation. Flower's key innovation lies in decomposing item-level rewards into constituent token rewards, enabling direct alignment between token generation probabilities and their reward signals. This mechanism achieves three critical advancements: (1) popularity bias mitigation and fairness enhancement through empirical distribution matching, (2) preservation of diversity through GFlowNet's proportional sampling, and (3) flexible integration of personalized preferences via adaptable token rewards. Experiments demonstrate Flower's superior distribution-fitting capability and its significant advantages over traditional SFT in terms of fairness, diversity, and accuracy, highlighting its potential to improve LLM-based recommendation systems. The implementation is available via https://github.com/Mr-Peach0301/Flower
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05139v2">Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 34 pages
    </div>
    <details class="paper-abstract">
      In this technical report, we tackle the challenges of training large-scale Mixture of Experts (MoE) models, focusing on overcoming cost inefficiency and resource limitations prevalent in such systems. To address these issues, we present two differently sized MoE large language models (LLMs), namely Ling-Lite and Ling-Plus (referred to as "Bailing" in Chinese, spelled B\v{a}il\'ing in Pinyin). Ling-Lite contains 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus boasts 290 billion parameters with 28.8 billion activated parameters. Both models exhibit comparable performance to leading industry benchmarks. This report offers actionable insights to improve the efficiency and accessibility of AI development in resource-constrained settings, promoting more scalable and sustainable technologies. Specifically, to reduce training costs for large-scale MoE models, we propose innovative methods for (1) optimization of model architecture and training processes, (2) refinement of training anomaly handling, and (3) enhancement of model evaluation efficiency. Additionally, leveraging high-quality data generated from knowledge graphs, our models demonstrate superior capabilities in tool use compared to other models. Ultimately, our experimental findings demonstrate that a 300B MoE LLM can be effectively trained on lower-performance devices while achieving comparable performance to models of a similar scale, including dense and MoE models. Compared to high-performance devices, utilizing a lower-specification hardware system during the pre-training phase demonstrates significant cost savings, reducing computing costs by approximately 20%. The models can be accessed at https://huggingface.co/inclusionAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06664v1">Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 14 pages, 1 main figure, 3 plots, Published at ICLR 2025 Workshop on Foundation Models in the Wild
    </div>
    <details class="paper-abstract">
      High-quality, error-free datasets are a key ingredient in building reliable, accurate, and unbiased machine learning (ML) models. However, real world datasets often suffer from errors due to sensor malfunctions, data entry mistakes, or improper data integration across multiple sources that can severely degrade model performance. Detecting and correcting these issues typically require tailor-made solutions and demand extensive domain expertise. Consequently, automation is challenging, rendering the process labor-intensive and tedious. In this study, we investigate whether Large Language Models (LLMs) can help alleviate the burden of manual data cleaning. We set up an experiment in which an LLM, paired with Python, is tasked with cleaning the training dataset to improve the performance of a learning algorithm without having the ability to modify the training pipeline or perform any feature engineering. We run this experiment on multiple Kaggle datasets that have been intentionally corrupted with errors. Our results show that LLMs can identify and correct erroneous entries, such as illogical values or outlier, by leveraging contextual information from other features within the same row, as well as feedback from previous iterations. However, they struggle to detect more complex errors that require understanding data distribution across multiple rows, such as trends and biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06646v1">Evaluating and Aligning Human Economic Risk Preferences in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in decision-making scenarios that involve risk assessment, yet their alignment with human economic rationality remains unclear. In this study, we investigate whether LLMs exhibit risk preferences consistent with human expectations across different personas. Specifically, we assess whether LLM-generated responses reflect appropriate levels of risk aversion or risk-seeking behavior based on individual's persona. Our results reveal that while LLMs make reasonable decisions in simplified, personalized risk contexts, their performance declines in more complex economic decision-making tasks. To address this, we propose an alignment method designed to enhance LLM adherence to persona-specific risk preferences. Our approach improves the economic rationality of LLMs in risk-related applications, offering a step toward more human-aligned AI decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12899v2">DreamStory: Open-Domain Story Visualization by LLM-Guided Multi-Subject Consistent Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Story visualization aims to create visually compelling images or videos corresponding to textual narratives. Despite recent advances in diffusion models yielding promising results, existing methods still struggle to create a coherent sequence of subject-consistent frames based solely on a story. To this end, we propose DreamStory, an automatic open-domain story visualization framework by leveraging the LLMs and a novel multi-subject consistent diffusion model. DreamStory consists of (1) an LLM acting as a story director and (2) an innovative Multi-Subject consistent Diffusion model (MSD) for generating consistent multi-subject across the images. First, DreamStory employs the LLM to generate descriptive prompts for subjects and scenes aligned with the story, annotating each scene's subjects for subsequent subject-consistent generation. Second, DreamStory utilizes these detailed subject descriptions to create portraits of the subjects, with these portraits and their corresponding textual information serving as multimodal anchors (guidance). Finally, the MSD uses these multimodal anchors to generate story scenes with consistent multi-subject. Specifically, the MSD includes Masked Mutual Self-Attention (MMSA) and Masked Mutual Cross-Attention (MMCA) modules. MMSA and MMCA modules ensure appearance and semantic consistency with reference images and text, respectively. Both modules employ masking mechanisms to prevent subject blending. To validate our approach and promote progress in story visualization, we established a benchmark, DS-500, which can assess the overall performance of the story visualization framework, subject-identification accuracy, and the consistency of the generation model. Extensive experiments validate the effectiveness of DreamStory in both subjective and objective evaluations. Please visit our project homepage at https://dream-xyz.github.io/dreamstory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13511v2">Slice-Level Scheduling for High Throughput and Load Balanced LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) iteratively generate text token by token, with memory usage increasing with the length of generated token sequences. Since the request generation length is generally unpredictable, it is difficult to estimate the time and memory required to process requests, thus posing a challenge for effective request scheduling. Conventional sequence-level scheduling (SLS) serves requests in a first-come first-served (FCFS) manner with static batching where requests with short generation lengths are delayed until those with long ones have finished generation. Besides, to avoid out-of-memory (OOM) errors, SLS batches requests using a small batch size, which limits throughput. Recently proposed iteration-level scheduling (ILS) improves this with continuous batching, timely completing requests and dynamically adding new ones, but often limits the number of parallel-processing requests to OOM errors, thus compromising throughput. Moreover, both SLS and ILS fail to effectively balance workload across multiple LLM instances. To tackle these challenges, we propose slice-level scheduling (SCLS). By splitting the predefined maximal generation length limit into slices and serving batches slice by slice, it provides a precise range of serving time and memory usage for batched requests, laying the foundation for effective scheduling. Experiments confirm that compared with SLS and ILS schedulers, SCLS can improve throughput by up to 315.8% and greatly mitigate load imbalance with proposed batching and offloading algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03594v2">Small but Mighty: Enhancing Time Series Forecasting with Lightweight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 20 pages, 10 figures
    </div>
    <details class="paper-abstract">
      While LLMs have demonstrated remarkable potential in time series forecasting, their practical deployment remains constrained by excessive computational demands and memory footprints. Existing LLM-based approaches typically suffer from three critical limitations: Inefficient parameter utilization in handling numerical time series patterns; Modality misalignment between continuous temporal signals and discrete text embeddings; and Inflexibility for real-time expert knowledge integration. We present SMETimes, the first systematic investigation of sub-3B parameter SLMs for efficient and accurate time series forecasting. Our approach centers on three key innovations: A statistically-enhanced prompting mechanism that bridges numerical time series with textual semantics through descriptive statistical features; A adaptive fusion embedding architecture that aligns temporal patterns with language model token spaces through learnable parameters; And a dynamic mixture-of-experts framework enabled by SLMs' computational efficiency, adaptively combining base predictions with domain-specific models. Extensive evaluations across seven benchmark datasets demonstrate that our 3B-parameter SLM achieves state-of-the-art performance on five primary datasets while maintaining 3.8x faster training and 5.2x lower memory consumption compared to 7B-parameter LLM baselines. Notably, the proposed model exhibits better learning capabilities, achieving 12.3% lower MSE than conventional LLM. Ablation studies validate that our statistical prompting and cross-modal fusion modules respectively contribute 15.7% and 18.2% error reduction in long-horizon forecasting tasks. By redefining the efficiency-accuracy trade-off landscape, this work establishes SLMs as viable alternatives to resource-intensive LLMs for practical time series forecasting. Code and models are available at https://github.com/xiyan1234567/SMETimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06550v1">BingoGuard: LLM Content Moderation Tools with Risk Levels</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 10 pages, 4 figures, 4 tables. ICLR 2025 poster
    </div>
    <details class="paper-abstract">
      Malicious content generated by large language models (LLMs) can pose varying degrees of harm. Although existing LLM-based moderators can detect harmful content, they struggle to assess risk levels and may miss lower-risk outputs. Accurate risk assessment allows platforms with different safety thresholds to tailor content filtering and rejection. In this paper, we introduce per-topic severity rubrics for 11 harmful topics and build BingoGuard, an LLM-based moderation system designed to predict both binary safety labels and severity levels. To address the lack of annotations on levels of severity, we propose a scalable generate-then-filter framework that first generates responses across different severity levels and then filters out low-quality responses. Using this framework, we create BingoGuardTrain, a training dataset with 54,897 examples covering a variety of topics, response severity, styles, and BingoGuardTest, a test set with 988 examples explicitly labeled based on our severity rubrics that enables fine-grained analysis on model behaviors on different severity levels. Our BingoGuard-8B, trained on BingoGuardTrain, achieves the state-of-the-art performance on several moderation benchmarks, including WildGuardTest and HarmBench, as well as BingoGuardTest, outperforming best public models, WildGuard, by 4.3\%. Our analysis demonstrates that incorporating severity levels into training significantly enhances detection performance and enables the model to effectively gauge the severity of harmful responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19038v2">DIESEL -- Dynamic Inference-Guidance via Evasion of Semantic Embeddings in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have had great success in tasks such as casual conversation, contributing to significant advancements in domains like virtual assistance. However, they often generate responses that are not aligned with human values (e.g., ethical standards, safety), leading to potentially unsafe or inappropriate outputs. While several techniques have been proposed to address this problem, they come with a cost, requiring computationally expensive training or dramatically increasing the inference time. In this paper, we present DIESEL, a lightweight inference-guidance technique that can be seamlessly integrated into any autoregressive LLM to semantically filter undesired concepts from the response. DIESEL can function either as a standalone safeguard or as an additional layer of defense, enhancing response safety by reranking the LLM's proposed tokens based on their similarity to predefined negative concepts in the latent space. Our evaluation demonstrates DIESEL's effectiveness on state-of-the-art conversational models, even in adversarial jailbreaking scenarios that challenge response safety. We also highlight DIESEL's generalization capabilities, showing that it can be used in use cases other than safety, providing general-purpose response filtering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07516v2">Exploring and Lifting the Robustness of LLM-powered Automated Program Repair with Metamorphic Testing</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      In recent years, Large language model-powered Automated Program Repair (LAPR) techniques have achieved state-of-the-art bug-fixing performance and have been pervasively applied and studied in both industry and academia. Nonetheless, LLMs were proved to be highly sensitive to input prompts, with slight differences in the expressions of semantically equivalent programs potentially causing repair failures. Therefore, it is crucial to conduct robustness testing on LAPR techniques before their practical deployment. However, related research is scarce. To this end, we propose MT-LAPR, a Metamorphic Testing framework exclusively for LAPR techniques, which summarizes nine widely-recognized Metamorphic Relations (MRs) by developers across three perturbation levels: token, statement, and block. Afterward, our proposed MRs are applied to buggy codes to generate test cases, which are semantically equivalent yet to affect the inference of LAPR. Experiments are carried out on two extensively examined bug-fixing datasets, i.e., Defect4J and QuixBugs, and four bug-fixing abled LLMs released recently, demonstrating that 34.4% - 48.5% of the test cases expose the instability of LAPR techniques on average, showing the effectiveness of MT-LAPR and uncovering a positive correlation between code readability and the robustness of LAPR techniques. Inspired by the above findings, this paper uses the test cases generated by MT-LAPR as samples to train a CodeT5-based code editing model aiming at improving code readability and then embeds it into the LAPR workflow as a data preprocessing step. Extensive experiments demonstrate that this approach significantly enhances the robustness of LAPR by 49.32% at most.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02930v2">Video LLMs for Temporal Reasoning in Long Videos</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      This paper introduces TemporalVLM, a video large language model (video LLM) capable of effective temporal reasoning and fine-grained understanding in long videos. At the core, our approach includes a visual encoder for mapping a long-term input video into features which are time-aware and contain both local and global cues. In particular, it first divides the input video into short-term clips, which are jointly encoded with their timestamps into time-sensitive local features. Next, the local features are passed through a bidirectional long short-term memory (BiLSTM) module for global feature aggregation. The extracted time-aware and multi-level features are important for accurate temporal reasoning and fine-grained understanding in long videos. Moreover, to facilitate the evaluation of TemporalVLM, we present a large-scale long video dataset of industry assembly processes, namely IndustryASM, which consists of videos recorded on factory floors with actions and timestamps annotated by industrial engineers for time and motion studies and temporal action segmentation evaluation. Finally, extensive experiments on datasets of long videos, including TimeIT and IndustryASM, show that TemporalVLM achieves superior performance than previous methods across temporal reasoning and fine-grained understanding tasks, namely dense video captioning, temporal video grounding, video highlight detection, and temporal action segmentation. To the best of our knowledge, our work is the first to incorporate LSTMs into video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15594v5">A Survey on LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Project Page: https://awesome-llm-as-a-judge.github.io/
    </div>
    <details class="paper-abstract">
      Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse domains, leading to the emergence of "LLM-as-a-Judge," where LLMs are employed as evaluators for complex tasks. With their ability to process diverse data types and provide scalable, cost-effective, and consistent assessments, LLMs present a compelling alternative to traditional expert-driven evaluations. However, ensuring the reliability of LLM-as-a-Judge systems remains a significant challenge that requires careful design and standardization. This paper provides a comprehensive survey of LLM-as-a-Judge, addressing the core question: How can reliable LLM-as-a-Judge systems be built? We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios. Additionally, we propose methodologies for evaluating the reliability of LLM-as-a-Judge systems, supported by a novel benchmark designed for this purpose. To advance the development and real-world deployment of LLM-as-a-Judge systems, we also discussed practical applications, challenges, and future directions. This survey serves as a foundational reference for researchers and practitioners in this rapidly evolving field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06433v1">Seesaw: High-throughput LLM Inference via Model Re-sharding</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      To improve the efficiency of distributed large language model (LLM) inference, various parallelization strategies, such as tensor and pipeline parallelism, have been proposed. However, the distinct computational characteristics inherent in the two stages of LLM inference-prefilling and decoding-render a single static parallelization strategy insufficient for the effective optimization of both stages. In this work, we present Seesaw, an LLM inference engine optimized for throughput-oriented tasks. The key idea behind Seesaw is dynamic model re-sharding, a technique that facilitates the dynamic reconfiguration of parallelization strategies across stages, thereby maximizing throughput at both phases. To mitigate re-sharding overhead and optimize computational efficiency, we employ tiered KV cache buffering and transition-minimizing scheduling. These approaches work synergistically to reduce the overhead caused by frequent stage transitions while ensuring maximum batching efficiency. Our evaluation demonstrates that Seesaw achieves a throughput increase of up to 1.78x (1.36x on average) compared to vLLM, the most widely used state-of-the-art LLM inference engine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06430v1">Graph Retrieval-Augmented LLM for Conversational Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Accepted by PAKDD 2025
    </div>
    <details class="paper-abstract">
      Conversational Recommender Systems (CRSs) have emerged as a transformative paradigm for offering personalized recommendations through natural language dialogue. However, they face challenges with knowledge sparsity, as users often provide brief, incomplete preference statements. While recent methods have integrated external knowledge sources to mitigate this, they still struggle with semantic understanding and complex preference reasoning. Recent Large Language Models (LLMs) demonstrate promising capabilities in natural language understanding and reasoning, showing significant potential for CRSs. Nevertheless, due to the lack of domain knowledge, existing LLM-based CRSs either produce hallucinated recommendations or demand expensive domain-specific training, which largely limits their applicability. In this work, we present G-CRS (Graph Retrieval-Augmented Large Language Model for Conversational Recommender Systems), a novel training-free framework that combines graph retrieval-augmented generation and in-context learning to enhance LLMs' recommendation capabilities. Specifically, G-CRS employs a two-stage retrieve-and-recommend architecture, where a GNN-based graph reasoner first identifies candidate items, followed by Personalized PageRank exploration to jointly discover potential items and similar user interactions. These retrieved contexts are then transformed into structured prompts for LLM reasoning, enabling contextually grounded recommendations without task-specific training. Extensive experiments on two public datasets show that G-CRS achieves superior recommendation performance compared to existing methods without requiring task-specific training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09187v2">GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language model (LLM) agents has raised new concerns regarding their safety and security, which cannot be addressed by traditional textual-harm-focused LLM guardrails. We propose GuardAgent, the first guardrail agent to protect the target agents by dynamically checking whether their actions satisfy given safety guard requests. Specifically, GuardAgent first analyzes the safety guard requests to generate a task plan, and then maps this plan into guardrail code for execution. By performing the code execution, GuardAgent can deterministically follow the safety guard request and safeguard target agents. In both steps, an LLM is utilized as the reasoning component, supplemented by in-context demonstrations retrieved from a memory module storing experiences from previous tasks. GuardAgent can understand different safety guard requests and provide reliable code-based guardrails with high flexibility and low operational overhead. In addition, we propose two novel benchmarks: EICU-AC benchmark to assess the access control for healthcare agents and Mind2Web-SC benchmark to evaluate the safety policies for web agents. We show that GuardAgent effectively moderates the violation actions for different types of agents on these two benchmarks with over 98% and 83% guardrail accuracies, respectively. Project page: https://guardagent.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06424v1">Training LLM-based Tutors to Improve Student Learning Outcomes in Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Generative artificial intelligence (AI) has the potential to scale up personalized tutoring through large language models (LLMs). Recent AI tutors are adapted for the tutoring task by training or prompting LLMs to follow effective pedagogical principles, though they are not trained to maximize student learning throughout the course of a dialogue. Therefore, they may engage with students in a suboptimal way. We address this limitation by introducing an approach to train LLMs to generate tutor utterances that maximize the likelihood of student correctness, while still encouraging the model to follow good pedagogical practice. Specifically, we generate a set of candidate tutor utterances and score them using (1) an LLM-based student model to predict the chance of correct student responses and (2) a pedagogical rubric evaluated by GPT-4o. We then use the resulting data to train an open-source LLM, Llama 3.1 8B, using direct preference optimization. We show that tutor utterances generated by our model lead to significantly higher chances of correct student responses while maintaining the pedagogical quality of GPT-4o. We also conduct qualitative analyses and a human evaluation to demonstrate that our model generates high quality tutor utterances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06410v1">Performant LLM Agentic Framework for Conversational AI</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The rise of Agentic applications and automation in the Voice AI industry has led to an increased reliance on Large Language Models (LLMs) to navigate graph-based logic workflows composed of nodes and edges. However, existing methods face challenges such as alignment errors in complex workflows and hallucinations caused by excessive context size. To address these limitations, we introduce the Performant Agentic Framework (PAF), a novel system that assists LLMs in selecting appropriate nodes and executing actions in order when traversing complex graphs. PAF combines LLM-based reasoning with a mathematically grounded vector scoring mechanism, achieving both higher accuracy and reduced latency. Our approach dynamically balances strict adherence to predefined paths with flexible node jumps to handle various user inputs efficiently. Experiments demonstrate that PAF significantly outperforms baseline methods, paving the way for scalable, real-time Conversational AI systems in complex business environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18966v2">Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 3 tables and 1 figures in the main text. This paper is accepted by NAACL 2025 findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated great performance across various benchmarks, showing potential as general-purpose task solvers. However, as LLMs are typically trained on vast amounts of data, a significant concern in their evaluation is data contamination, where overlap between training data and evaluation datasets inflates performance assessments. Multiple approaches have been developed to identify data contamination. These approaches rely on specific assumptions that may not hold universally across different settings. To bridge this gap, we systematically review 50 papers on data contamination detection, categorize the underlying assumptions, and assess whether they have been rigorously validated. We identify and analyze eight categories of assumptions and test three of them as case studies. Our case studies focus on detecting direct, instance-level data contamination, which is also referred to as Membership Inference Attacks (MIA). Our analysis reveals that MIA approaches based on these three assumptions can have similar performance to random guessing, on datasets used in LLM pretraining, suggesting that current LLMs might learn data distributions rather than memorizing individual instances. Meanwhile, MIA can easily fail when there are data distribution shifts between the seen and unseen instances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06394v1">How LLMs Learn: Tracing Internal Representations with Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Our code, demo, SAE weights are available at: https://github.com/llm-jp/llm-jp-sae
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable multilingual capabilities and broad knowledge. However, the internal mechanisms underlying the development of these capabilities remain poorly understood. To investigate this, we analyze how the information encoded in LLMs' internal representations evolves during the training process. Specifically, we train sparse autoencoders at multiple checkpoints of the model and systematically compare the interpretative results across these stages. Our findings suggest that LLMs initially acquire language-specific knowledge independently, followed by cross-linguistic correspondences. Moreover, we observe that after mastering token-level knowledge, the model transitions to learning higher-level, abstract concepts, indicating the development of more conceptual understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06362v1">Adaptive Audio-Visual Speech Recognition via Matryoshka-Based Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Audio-Visual Speech Recognition (AVSR) leverages both audio and visual modalities to enhance speech recognition robustness, particularly in noisy environments. Recent advancements in Large Language Models (LLMs) have demonstrated their effectiveness in speech recognition, including AVSR. However, due to the significant length of speech representations, direct integration with LLMs imposes substantial computational costs. Prior approaches address this by compressing speech representations before feeding them into LLMs. However, higher compression ratios often lead to performance degradation, necessitating a trade-off between computational efficiency and recognition accuracy. To address this challenge, we propose Llama-MTSK, the first Matryoshka-based Multimodal LLM for AVSR, which enables flexible adaptation of the audio-visual token allocation based on specific computational constraints while preserving high performance. Our approach, inspired by Matryoshka Representation Learning, encodes audio-visual representations at multiple granularities within a single model, eliminating the need to train separate models for different compression levels. Moreover, to efficiently fine-tune the LLM, we introduce three LoRA-based Matryoshka strategies using global and scale-specific LoRA modules. Extensive evaluations on the two largest AVSR datasets demonstrate that Llama-MTSK achieves state-of-the-art results, matching or surpassing models trained independently at fixed compression levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07670v1">Retrieval Augmented Generation with Multi-Modal LLM Framework for Wireless Environments</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Accepted @ ICC 2025
    </div>
    <details class="paper-abstract">
      Future wireless networks aim to deliver high data rates and lower power consumption while ensuring seamless connectivity, necessitating robust optimization. Large language models (LLMs) have been deployed for generalized optimization scenarios. To take advantage of generative AI (GAI) models, we propose retrieval augmented generation (RAG) for multi-sensor wireless environment perception. Utilizing domain-specific prompt engineering, we apply RAG to efficiently harness multimodal data inputs from sensors in a wireless environment. Key pre-processing pipelines including image-to-text conversion, object detection, and distance calculations for multimodal RAG input from multi-sensor data are proposed to obtain a unified vector database crucial for optimizing LLMs in global wireless tasks. Our evaluation, conducted with OpenAI's GPT and Google's Gemini models, demonstrates an 8%, 8%, 10%, 7%, and 12% improvement in relevancy, faithfulness, completeness, similarity, and accuracy, respectively, compared to conventional LLM-based designs. Furthermore, our RAG-based LLM framework with vectorized databases is computationally efficient, providing real-time convergence under latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08704v1">Life-Cycle Routing Vulnerabilities of LLM Router</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in natural language processing, yet their performance and computational costs vary significantly. LLM routers play a crucial role in dynamically balancing these trade-offs. While previous studies have primarily focused on routing efficiency, security vulnerabilities throughout the entire LLM router life cycle, from training to inference, remain largely unexplored. In this paper, we present a comprehensive investigation into the life-cycle routing vulnerabilities of LLM routers. We evaluate both white-box and black-box adversarial robustness, as well as backdoor robustness, across several representative routing models under extensive experimental settings. Our experiments uncover several key findings: 1) Mainstream DNN-based routers tend to exhibit the weakest adversarial and backdoor robustness, largely due to their strong feature extraction capabilities that amplify vulnerabilities during both training and inference; 2) Training-free routers demonstrate the strongest robustness across different attack types, benefiting from the absence of learnable parameters that can be manipulated. These findings highlight critical security risks spanning the entire life cycle of LLM routers and provide insights for developing more robust models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06792v1">On the Mutual Influence of Gender and Occupation in LLM Representations</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 In submission
    </div>
    <details class="paper-abstract">
      We examine LLM representations of gender for first names in various occupational contexts to study how occupations and the gender perception of first names in LLMs influence each other mutually. We find that LLMs' first-name gender representations correlate with real-world gender statistics associated with the name, and are influenced by the co-occurrence of stereotypically feminine or masculine occupations. Additionally, we study the influence of first-name gender representations on LLMs in a downstream occupation prediction task and their potential as an internal metric to identify extrinsic model biases. While feminine first-name embeddings often raise the probabilities for female-dominated jobs (and vice versa for male-dominated jobs), reliably using these internal gender representations for bias detection remains challenging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06791v1">AutoMisty: A Multi-Agent LLM Framework for Automated Code Generation in the Misty Social Robot</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      The social robot's open API allows users to customize open-domain interactions. However, it remains inaccessible to those without programming experience. In this work, we introduce AutoMisty, the first multi-agent collaboration framework powered by large language models (LLMs), to enable the seamless generation of executable Misty robot code from natural language instructions. AutoMisty incorporates four specialized agent modules to manage task decomposition, assignment, problem-solving, and result synthesis. Each agent incorporates a two-layer optimization mechanism, with self-reflection for iterative refinement and human-in-the-loop for better alignment with user preferences. AutoMisty ensures a transparent reasoning process, allowing users to iteratively refine tasks through natural language feedback for precise execution. To evaluate AutoMisty's effectiveness, we designed a benchmark task set spanning four levels of complexity and conducted experiments in a real Misty robot environment. Extensive evaluations demonstrate that AutoMisty not only consistently generates high-quality code but also enables precise code control, significantly outperforming direct reasoning with ChatGPT-4o and ChatGPT-o1. All code, optimized APIs, and experimental videos will be publicly released through the webpage: https://wangxiaoshawn.github.io/AutoMisty.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01001v2">Towards An Efficient LLM Training Paradigm for CTR Prediction</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated tremendous potential as the next-generation ranking-based recommendation system. Many recent works have shown that LLMs can significantly outperform conventional click-through-rate (CTR) prediction approaches. Despite such promising results, the computational inefficiency inherent in the current training paradigm makes it particularly challenging to train LLMs for ranking-based recommendation tasks on large datasets. To train LLMs for CTR prediction, most existing studies adopt the prevalent ''sliding-window'' paradigm. Given a sequence of $m$ user interactions, a unique training prompt is constructed for each interaction by designating it as the prediction target along with its preceding $n$ interactions serving as context. In turn, the sliding-window paradigm results in an overall complexity of $O(mn^2)$ that scales linearly with the length of user interactions. Consequently, a direct adoption to train LLMs with such strategy can result in prohibitively high training costs as the length of interactions grows. To alleviate the computational inefficiency, we propose a novel training paradigm, namely Dynamic Target Isolation (DTI), that structurally parallelizes the training of $k$ (where $k >> 1$) target interactions. Furthermore, we identify two major bottlenecks - hidden-state leakage and positional bias overfitting - that limit DTI to only scale up to a small value of $k$ (e.g., 5) then propose a computationally light solution to effectively tackle each. Through extensive experiments on three widely adopted public CTR datasets, we empirically show that DTI reduces training time by an average of $\textbf{92%}$ (e.g., from $70.5$ hrs to $5.31$ hrs), without compromising CTR prediction performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06781v1">Dr Genre: Reinforcement Learning from Decoupled LLM Feedback for Generic Text Rewriting</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 29 pages, 4 figures, 25 tables
    </div>
    <details class="paper-abstract">
      Generic text rewriting is a prevalent large language model (LLM) application that covers diverse real-world tasks, such as style transfer, fact correction, and email editing. These tasks vary in rewriting objectives (e.g., factual consistency vs. semantic preservation), making it challenging to develop a unified model that excels across all dimensions. Existing methods often specialize in either a single task or a specific objective, limiting their generalizability. In this work, we introduce a generic model proficient in factuality, stylistic, and conversational rewriting tasks. To simulate real-world user rewrite requests, we construct a conversational rewrite dataset, ChatRewrite, that presents ``natural''-sounding instructions, from raw emails using LLMs. Combined with other popular rewrite datasets, including LongFact for the factuality rewrite task and RewriteLM for the stylistic rewrite task, this forms a broad benchmark for training and evaluating generic rewrite models. To align with task-specific objectives, we propose Dr Genre, a Decoupled-reward learning framework for Generic rewriting, that utilizes objective-oriented reward models with a task-specific weighting. Evaluation shows that \approach delivers higher-quality rewrites across all targeted tasks, improving objectives including instruction following (agreement), internal consistency (coherence), and minimal unnecessary edits (conciseness).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06689v1">DependEval: Benchmarking LLMs for Repository Dependency Understanding</a></div>
    <div class="paper-meta">
      📅 2025-03-09
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown considerable promise in code generation, real-world software development demands advanced repository-level reasoning. This includes understanding dependencies, project structures, and managing multi-file changes. However, the ability of LLMs to effectively comprehend and handle complex code repositories has yet to be fully explored. To address challenges, we introduce a hierarchical benchmark designed to evaluate repository dependency understanding (DependEval). Benchmark is based on 15,576 repositories collected from real-world websites. It evaluates models on three core tasks: Dependency Recognition, Repository Construction, and Multi-file Editing, across 8 programming languages from actual code repositories. Our evaluation of over 25 LLMs reveals substantial performance gaps and provides valuable insights into repository-level code understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13802v2">Enhancing LLMs for Governance with Human Oversight: Evaluating and Aligning LLMs on Expert Classification of Climate Misinformation for Detecting False or Misleading Claims about Climate Change</a></div>
    <div class="paper-meta">
      📅 2025-03-09
      | 💬 International Workshop on AI Governance: Alignment, Morality and Law (AIGOV) 2025. AAAI Conference on Artificial Intelligence
    </div>
    <details class="paper-abstract">
      Climate misinformation is a problem that has the potential to be substantially aggravated by the development of Large Language Models (LLMs). In this study we evaluate the potential for LLMs to be part of the solution for mitigating online dis/misinformation rather than the problem. Employing a public expert annotated dataset and a curated sample of social media content we evaluate the performance of proprietary vs. open source LLMs on climate misinformation classification task, comparing them to existing climate-focused computer-assisted tools and expert assessments. Results show (1) open-source models substantially under-perform in classifying climate misinformation compared to proprietary models, (2) existing climate-focused computer-assisted tools leveraging expert-annotated datasets continues to outperform many of proprietary models, including GPT-4o, and (3) demonstrate the efficacy and generalizability of fine-tuning GPT-3.5-turbo on expert annotated dataset in classifying claims about climate change at the equivalency of climate change experts with over 20 years of experience in climate communication. These findings highlight 1) the importance of incorporating human-oversight, such as incorporating expert-annotated datasets in training LLMs, for governance tasks that require subject-matter expertise like classifying climate misinformation, and 2) the potential for LLMs in facilitating civil society organizations to engage in various governance tasks such as classifying false or misleading claims in domains beyond climate change such as politics and health science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06139v1">GRP: Goal-Reversed Prompting for Zero-Shot Evaluation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Ongoing Work
    </div>
    <details class="paper-abstract">
      Using Large Language Models (LLMs) to evaluate and compare two answers from different models typically involves having LLM-based judges select the better answer. However, humans often approach problem-solving from a reverse perspective, for instance, by choosing the worse option instead of the better one in a pairwise comparison. Generally, this kind of reverse thinking plays a crucial role in human reasoning and decision-making and can further test the difference between original and reverse thought processes simultaneously. To address the above issue, in this paper, we propose a Goal-Reversed Prompting (GRP) approach for pairwise evaluation that shifts the original task from selecting the better answer to choosing the worse one. We encourage LLMs to think in reverse by prompting LLMs to identify the worse response. Experiments on closed-source models demonstrate that GRP significantly enhances evaluation capabilities, outperforming the prompt template with the original goal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06119v1">Unlocking Pretrained LLMs for Motion-Related Multimodal Generation: A Fine-Tuning Approach to Unify Diffusion and Next-Token Prediction</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      In this paper, we propose a unified framework that leverages a single pretrained LLM for Motion-related Multimodal Generation, referred to as MoMug. MoMug integrates diffusion-based continuous motion generation with the model's inherent autoregressive discrete text prediction capabilities by fine-tuning a pretrained LLM. This enables seamless switching between continuous motion output and discrete text token prediction within a single model architecture, effectively combining the strengths of both diffusion- and LLM-based approaches. Experimental results show that, compared to the most recent LLM-based baseline, MoMug improves FID by 38% and mean accuracy across seven metrics by 16.61% on the text-to-motion task. Additionally, it improves mean accuracy across eight metrics by 8.44% on the text-to-motion task. To the best of our knowledge, this is the first approach to integrate diffusion- and LLM-based generation within a single model for motion-related multimodal tasks while maintaining low training costs. This establishes a foundation for future advancements in motion-related generation, paving the way for high-quality yet cost-efficient motion synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01600v3">Reinforcement Learning for Long-Horizon Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Interactive digital agents (IDAs) leverage APIs of stateful digital environments to perform tasks in response to user requests. While IDAs powered by instruction-tuned large language models (LLMs) can react to feedback from interface invocations in multi-step exchanges, they have not been trained in their respective digital environments. Prior methods accomplish less than half of tasks in sophisticated benchmarks such as AppWorld. We present a reinforcement learning (RL) approach that trains IDAs directly in their target environments. We formalize this training as a partially observable Markov decision process and derive LOOP, a data- and memory-efficient variant of proximal policy optimization. LOOP uses no value network and maintains exactly one copy of the underlying LLM in memory, making its implementation straightforward and as memory-efficient as fine-tuning a single LLM. A 32-billion-parameter agent trained with LOOP in the AppWorld environment outperforms the much larger OpenAI o1 agent by 9 percentage points (15% relative). To our knowledge, this is the first reported application of RL to IDAs that interact with a stateful, multi-domain, multi-app environment via direct API calls. Our analysis sheds light on the effectiveness of RL in this area, showing that the agent learns to consult the API documentation, avoid unwarranted assumptions, minimize confabulation, and recover from setbacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02644v2">Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10868v3">NitiBench: A Comprehensive Study of LLM Framework Capabilities for Thai Legal Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the legal domain holds significant potential for information retrieval and question answering, yet Thai legal QA systems face challenges due to a lack of standardized evaluation benchmarks and the complexity of Thai legal structures. This paper introduces NitiBench, a benchmark comprising two datasets: the NitiBench-CCL, covering general Thai financial law, and the NitiBench-Tax, which includes real-world tax law cases requiring advanced legal reasoning. We evaluate retrieval-augmented generation (RAG) and long-context LLM-based approaches to address three key research questions: the impact of domain-specific components like section-based chunking and cross-referencing, the comparative performance of different retrievers and LLMs, and the viability of long-context LLMs as an alternative to RAG. Our results show that section-based chunking significantly improves retrieval and end-to-end performance, current retrievers struggle with complex queries, and long-context LLMs still underperform RAG-based systems in Thai legal QA. To support fair evaluation, we propose tailored multi-label retrieval metrics and the use of an LLM-as-judge for coverage and contradiction detection method. These findings highlight the limitations of current Thai legal NLP solutions and provide a foundation for future research in the field. We also open-sourced our codes and dataset to available publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06054v1">Fine-Grained Bias Detection in LLM: Enhancing detection mechanisms for nuanced biases</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Bias detection, Large Language Models, nuanced biases, fine-grained mechanisms, model transparency, ethical AI
    </div>
    <details class="paper-abstract">
      Recent advancements in Artificial Intelligence, particularly in Large Language Models (LLMs), have transformed natural language processing by improving generative capabilities. However, detecting biases embedded within these models remains a challenge. Subtle biases can propagate misinformation, influence decision-making, and reinforce stereotypes, raising ethical concerns. This study presents a detection framework to identify nuanced biases in LLMs. The approach integrates contextual analysis, interpretability via attention mechanisms, and counterfactual data augmentation to capture hidden biases across linguistic contexts. The methodology employs contrastive prompts and synthetic datasets to analyze model behaviour across cultural, ideological, and demographic scenarios. Quantitative analysis using benchmark datasets and qualitative assessments through expert reviews validate the effectiveness of the framework. Results show improvements in detecting subtle biases compared to conventional methods, which often fail to highlight disparities in model responses to race, gender, and socio-political contexts. The framework also identifies biases arising from imbalances in training data and model architectures. Continuous user feedback ensures adaptability and refinement. This research underscores the importance of proactive bias mitigation strategies and calls for collaboration between policymakers, AI developers, and regulators. The proposed detection mechanisms enhance model transparency and support responsible LLM deployment in sensitive applications such as education, legal systems, and healthcare. Future work will focus on real-time bias monitoring and cross-linguistic generalization to improve fairness and inclusivity in AI-driven communication tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06047v1">DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 43 pages, 5 figures, conference
    </div>
    <details class="paper-abstract">
      Large Language Model~(LLM) based agents have been increasingly popular in solving complex and dynamic tasks, which requires proper evaluation systems to assess their capabilities. Nevertheless, existing benchmarks usually either focus on single-objective tasks or use overly broad assessing metrics, failing to provide a comprehensive inspection of the actual capabilities of LLM-based agents in complicated decision-making tasks. To address these issues, we introduce DSGBench, a more rigorous evaluation platform for strategic decision-making. Firstly, it incorporates six complex strategic games which serve as ideal testbeds due to their long-term and multi-dimensional decision-making demands and flexibility in customizing tasks of various difficulty levels or multiple targets. Secondly, DSGBench employs a fine-grained evaluation scoring system which examines the decision-making capabilities by looking into the performance in five specific dimensions and offering a comprehensive assessment in a well-designed way. Furthermore, DSGBench also incorporates an automated decision-tracking mechanism which enables in-depth analysis of agent behaviour patterns and the changes in their strategies. We demonstrate the advances of DSGBench by applying it to multiple popular LLM-based agents and our results suggest that DSGBench provides valuable insights in choosing LLM-based agents as well as improving their future development. DSGBench is available at https://github.com/DeciBrain-Group/DSGBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06040v1">Mitigating Memorization in LLMs using Activation Steering</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      The memorization of training data by Large Language Models (LLMs) poses significant risks, including privacy leaks and the regurgitation of copyrighted content. Activation steering, a technique that directly intervenes in model activations, has emerged as a promising approach for manipulating LLMs. In this work, we explore the effectiveness of activation steering in reducing memorization while preserving generalization capabilities. We conduct empirical evaluations using a controlled memorization benchmark of literary material and demonstrate that our method successfully suppresses memorized content with minimal degradation in model performance in Gemma. Additionally, we analyze the trade-offs between suppression effectiveness and linguistic fluency, highlighting the advantages and limitations of activation-based interventions. Our findings contribute to ongoing efforts in developing safer and more privacy-preserving LLMs by providing a practical and efficient mechanism to mitigate unintended memorization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06034v1">Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Rank-R1, a novel LLM-based reranker that performs reasoning over both the user query and candidate documents before performing the ranking task. Existing document reranking methods based on large language models (LLMs) typically rely on prompting or fine-tuning LLMs to order or label candidate documents according to their relevance to a query. For Rank-R1, we use a reinforcement learning algorithm along with only a small set of relevance labels (without any reasoning supervision) to enhance the reasoning ability of LLM-based rerankers. Our hypothesis is that adding reasoning capabilities to the rerankers can improve their relevance assessement and ranking capabilities. Our experiments on the TREC DL and BRIGHT datasets show that Rank-R1 is highly effective, especially for complex queries. In particular, we find that Rank-R1 achieves effectiveness on in-domain datasets at par with that of supervised fine-tuning methods, but utilizing only 18\% of the training data used by the fine-tuning methods. We also find that the model largely outperforms zero-shot and supervised fine-tuning when applied to out-of-domain datasets featuring complex queries, especially when a 14B-size model is used. Finally, we qualitatively observe that Rank-R1's reasoning process improves the explainability of the ranking results, opening new opportunities for search engine results presentation and fruition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06029v1">SmartBench: Is Your LLM Truly a Good Chinese Smartphone Assistant?</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to daily life, especially advancing as intelligent assistants through on-device deployment on smartphones. However, existing LLM evaluation benchmarks predominantly focus on objective tasks like mathematics and coding in English, which do not necessarily reflect the practical use cases of on-device LLMs in real-world mobile scenarios, especially for Chinese users. To address these gaps, we introduce SmartBench, the first benchmark designed to evaluate the capabilities of on-device LLMs in Chinese mobile contexts. We analyze functionalities provided by representative smartphone manufacturers and divide them into five categories: text summarization, text Q\&A, information extraction, content creation, and notification management, further detailed into 20 specific tasks. For each task, we construct high-quality datasets comprising 50 to 200 question-answer pairs that reflect everyday mobile interactions, and we develop automated evaluation criteria tailored for these tasks. We conduct comprehensive evaluations of on-device LLMs and MLLMs using SmartBench and also assess their performance after quantized deployment on real smartphone NPUs. Our contributions provide a standardized framework for evaluating on-device LLMs in Chinese, promoting further development and optimization in this critical area. Code and data will be available at https://github.com/Lucky-Lance/SmartBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00096v2">BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 8 main text pages, 5 main figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and LLM-based agents show great promise in accelerating scientific research. Existing benchmarks for measuring this potential and guiding future development continue to evolve from pure recall and rote knowledge tasks, towards more practical work such as literature review and experimental planning. Bioinformatics is a domain where fully autonomous AI-driven discovery may be near, but no extensive benchmarks for measuring progress have been introduced to date. We therefore present the Bioinformatics Benchmark (BixBench), a dataset comprising over 50 real-world scenarios of practical biological data analysis with nearly 300 associated open-answer questions designed to measure the ability of LLM-based agents to explore biological datasets, perform long, multi-step analytical trajectories, and interpret the nuanced results of those analyses. We evaluate the performance of two frontier LLMs (GPT-4o and Claude 3.5 Sonnet) using a custom agent framework we open source. We find that even the latest frontier models only achieve 17% accuracy in the open-answer regime, and no better than random in a multiple-choice setting. By exposing the current limitations of frontier models, we hope BixBench can spur the development of agents capable of conducting rigorous bioinformatic analysis and accelerate scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23252v3">Evaluating Cultural and Social Awareness of LLM Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) expand into performing as agents for real-world applications beyond traditional NLP tasks, evaluating their robustness becomes increasingly important. However, existing benchmarks often overlook critical dimensions like cultural and social awareness. To address these, we introduce CASA, a benchmark designed to assess LLM agents' sensitivity to cultural and social norms across two web-based tasks: online shopping and social discussion forums. Our approach evaluates LLM agents' ability to detect and appropriately respond to norm-violating user queries and observations. Furthermore, we propose a comprehensive evaluation framework that measures awareness coverage, helpfulness in managing user queries, and the violation rate when facing misleading web content. Experiments show that current LLMs perform significantly better in non-agent than in web-based agent environments, with agents achieving less than 10% awareness coverage and over 40% violation rates. To improve performance, we explore two methods: prompting and fine-tuning, and find that combining both methods can offer complementary advantages -- fine-tuning on culture-specific datasets significantly enhances the agents' ability to generalize across different regions, while prompting boosts the agents' ability to navigate complex tasks. These findings highlight the importance of constantly benchmarking LLM agents' cultural and social awareness during the development cycle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11721v2">Explain-Query-Test: Self-Evaluating LLMs Via Explanation and Comprehension Discrepancy</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Accepted to ICLR 2025, SSI-FM
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in generating detailed and coherent explanations of complex concepts. However, the extent to which these models truly comprehend the concepts they articulate remains unclear. To assess the level of comprehension of a model relative to the content it generates, we implemented a self-evaluation pipeline where models: (i) given a topic generate an excerpt with information about the topic, (ii) given an excerpt generate question-answer pairs, and finally (iii) given a question generate an answer. We refer to this self-evaluation approach as Explain-Query-Test (EQT). Interestingly, the accuracy on generated questions resulting from running the EQT pipeline correlates strongly with the model performance as verified by typical benchmarks such as MMLU-Pro. In other words, EQT's performance is predictive of MMLU-Pro's, and EQT can be used to rank models without the need for any external source of evaluation data other than lists of topics of interest. Moreover, our results reveal a disparity between the models' ability to produce detailed explanations and their performance on questions related to those explanations. This gap highlights fundamental limitations in the internal knowledge representation and reasoning abilities of current LLMs. We release the code at https://github.com/asgsaeid/EQT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06330v1">States of LLM-generated Texts and Phase Transitions between them</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Published as a conference paper at MathAI 2025
    </div>
    <details class="paper-abstract">
      It is known for some time that autocorrelations of words in human-written texts decay according to a power law. Recent works have also shown that the autocorrelations decay in texts generated by LLMs is qualitatively different from the literary texts. Solid state physics tie the autocorrelations decay laws to the states of matter. In this work, we empirically demonstrate that, depending on the temperature parameter, LLMs can generate text that can be classified as solid, critical state or gas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06327v1">Unveiling Inefficiencies in LLM-Generated Code: Toward a Comprehensive Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely adopted for automated code generation with promising results. Although prior research has assessed LLM-generated code and identified various quality issues -- such as redundancy, poor maintainability, and sub-optimal performance a systematic understanding and categorization of these inefficiencies remain unexplored. Without such knowledge, practitioners struggle to optimize LLM-generated code for real-world applications, limiting its adoption. This study can also guide improving code LLMs, enhancing the quality and efficiency of code generation. Therefore, in this study, we empirically investigate inefficiencies in LLM-generated code by state-of-the-art models, i.e., CodeLlama, DeepSeek-Coder, and CodeGemma. To do so, we analyze 492 generated code snippets in the HumanEval++ dataset. We then construct a taxonomy of inefficiencies in LLM-generated code that includes 5 categories General Logic, Performance, Readability, Maintainability, and Errors) and 19 subcategories of inefficiencies. We then validate the proposed taxonomy through an online survey with 58 LLM practitioners and researchers. Our study indicates that logic and performance-related inefficiencies are the most popular, relevant, and frequently co-occur and impact overall code quality inefficiency. Our taxonomy provides a structured basis for evaluating the quality LLM-generated code and guiding future research to improve code generation efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06313v1">Advancing Autonomous Vehicle Intelligence: Deep Learning and Multimodal LLM for Traffic Sign Recognition and Robust Lane Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Autonomous vehicles (AVs) require reliable traffic sign recognition and robust lane detection capabilities to ensure safe navigation in complex and dynamic environments. This paper introduces an integrated approach combining advanced deep learning techniques and Multimodal Large Language Models (MLLMs) for comprehensive road perception. For traffic sign recognition, we systematically evaluate ResNet-50, YOLOv8, and RT-DETR, achieving state-of-the-art performance of 99.8% with ResNet-50, 98.0% accuracy with YOLOv8, and achieved 96.6% accuracy in RT-DETR despite its higher computational complexity. For lane detection, we propose a CNN-based segmentation method enhanced by polynomial curve fitting, which delivers high accuracy under favorable conditions. Furthermore, we introduce a lightweight, Multimodal, LLM-based framework that directly undergoes instruction tuning using small yet diverse datasets, eliminating the need for initial pretraining. This framework effectively handles various lane types, complex intersections, and merging zones, significantly enhancing lane detection reliability by reasoning under adverse conditions. Despite constraints in available training resources, our multimodal approach demonstrates advanced reasoning capabilities, achieving a Frame Overall Accuracy (FRM) of 53.87%, a Question Overall Accuracy (QNS) of 82.83%, lane detection accuracies of 99.6% in clear conditions and 93.0% at night, and robust performance in reasoning about lane invisibility due to rain (88.4%) or road degradation (95.6%). The proposed comprehensive framework markedly enhances AV perception reliability, thus contributing significantly to safer autonomous driving across diverse and challenging road scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07923v2">Asking Again and Again: Exploring LLM Robustness to Repeated Questions</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      This study investigates whether repeating questions within prompts influences the performance of large language models (LLMs). We hypothesize that reiterating a question within a single prompt might enhance the model's focus on key elements of the query. We evaluate five recent LLMs -- including GPT-4o-mini, DeepSeek-V3, and smaller open-source models -- on three reading comprehension datasets under different prompt settings, varying question repetition levels (1, 3, or 5 times per prompt). Our results demonstrate that question repetition can increase models' accuracy by up to $6\%$. However, across all models, settings, and datasets, we do not find the result statistically significant. These findings provide insights into prompt design and LLM behavior, suggesting that repetition alone does not significantly impact output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06273v1">Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      We explore a novel zero-shot Audio-Visual Speech Recognition (AVSR) framework, dubbed Zero-AVSR, which enables speech recognition in target languages without requiring any audio-visual speech data in those languages. Specifically, we introduce the Audio-Visual Speech Romanizer (AV-Romanizer), which learns language-agnostic speech representations by predicting Roman text. Then, by leveraging the strong multilingual modeling capabilities of Large Language Models (LLMs), we propose converting the predicted Roman text into language-specific graphemes, forming the proposed Cascaded Zero-AVSR. Taking it a step further, we explore a unified Zero-AVSR approach by directly integrating the audio-visual speech representations encoded by the AV-Romanizer into the LLM. This is achieved through finetuning the adapter and the LLM using our proposed multi-task learning scheme. To capture the wide spectrum of phonetic and linguistic diversity, we also introduce a Multilingual Audio-Visual Romanized Corpus (MARC) consisting of 2,916 hours of audio-visual speech data across 82 languages, along with transcriptions in both language-specific graphemes and Roman text. Extensive analysis and experiments confirm that the proposed Zero-AVSR framework has the potential to expand language support beyond the languages seen during the training of the AV-Romanizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06253v1">MAD-MAX: Modular And Diverse Malicious Attack MiXtures for Automated LLM Red Teaming</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      With LLM usage rapidly increasing, their vulnerability to jailbreaks that create harmful outputs are a major security risk. As new jailbreaking strategies emerge and models are changed by fine-tuning, continuous testing for security vulnerabilities is necessary. Existing Red Teaming methods fall short in cost efficiency, attack success rate, attack diversity, or extensibility as new attack types emerge. We address these challenges with Modular And Diverse Malicious Attack MiXtures (MAD-MAX) for Automated LLM Red Teaming. MAD-MAX uses automatic assignment of attack strategies into relevant attack clusters, chooses the most relevant clusters for a malicious goal, and then combines strategies from the selected clusters to achieve diverse novel attacks with high attack success rates. MAD-MAX further merges promising attacks together at each iteration of Red Teaming to boost performance and introduces a similarity filter to prune out similar attacks for increased cost efficiency. The MAD-MAX approach is designed to be easily extensible with newly discovered attack strategies and outperforms the prominent Red Teaming method Tree of Attacks with Pruning (TAP) significantly in terms of Attack Success Rate (ASR) and queries needed to achieve jailbreaks. MAD-MAX jailbreaks 97% of malicious goals in our benchmarks on GPT-4o and Gemini-Pro compared to TAP with 66%. MAD-MAX does so with only 10.9 average queries to the target LLM compared to TAP with 23.3. WARNING: This paper contains contents which are offensive in nature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06705v2">LLMs can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Accepted by IJCAI 2024
    </div>
    <details class="paper-abstract">
      Self-correction is emerging as a promising approach to mitigate the issue of hallucination in Large Language Models (LLMs). To facilitate effective self-correction, recent research has proposed mistake detection as its initial step. However, current literature suggests that LLMs often struggle with reliably identifying reasoning mistakes when using simplistic prompting strategies. To address this challenge, we introduce a unique prompting strategy, termed the Pedagogical Chain-of-Thought (PedCoT), which is specifically designed to guide the identification of reasoning mistakes, particularly mathematical reasoning mistakes. PedCoT consists of pedagogical principles for prompts (PPP) design, two-stage interaction process (TIP) and grounded PedCoT prompts, all inspired by the educational theory of the Bloom Cognitive Model (BCM). We evaluate our approach on two public datasets featuring math problems of varying difficulty levels. The experiments demonstrate that our zero-shot prompting strategy significantly outperforms strong baselines. The proposed method can achieve the goal of reliable mathematical mistake identification and provide a foundation for automatic math answer grading. The results underscore the significance of educational theory, serving as domain knowledge, in guiding prompting strategy design for addressing challenging tasks with LLMs effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06203v1">Generation of Optimized Solidity Code for Machine Learning Models using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      While a plethora of machine learning (ML) models are currently available, along with their implementation on disparate platforms, there is hardly any verifiable ML code which can be executed on public blockchains. We propose a novel approach named LMST that enables conversion of the inferencing path of an ML model as well as its weights trained off-chain into Solidity code using Large Language Models (LLMs). Extensive prompt engineering is done to achieve gas cost optimization beyond mere correctness of the produced code, while taking into consideration the capabilities and limitations of the Ethereum Virtual Machine. We have also developed a proof of concept decentralized application using the code so generated for verifying the accuracy claims of the underlying ML model. An extensive set of experiments demonstrate the feasibility of deploying ML models on blockchains through automated code translation using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05673v5">Flow of Reasoning:Training LLMs for Divergent Problem Solving with Minimal Examples</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      The ability to generate diverse solutions to a given problem is a hallmark of human creativity. This divergent reasoning is also crucial for machines, enhancing their robustness and enabling them to assist humans in many applications such as scientific discovery. However, existing approaches to multi-step reasoning with large language models (LLMs) have mostly focused only on reasoning accuracy, without further discovering more diverse valid solutions. For example, supervised fine-tuning can improve LLM reasoning quality, but requires extensive supervised data to capture the full range of possible solutions. Reward-maximization reinforcement learning aims to find limited highest-reward solutions while neglecting the solution diversity. To fill this gap, we propose Flow of Reasoning (FoR), an efficient diversity-seeking LLM finetuning method aimed at improving reasoning quality and diversity with minimal data. FoR formulates multi-step LLM reasoning as a Markovian flow on a DAG-structured reasoning graph. This formulation allows us to incorporate and adapt principled GFlowNet approaches, for finetuning LLMs to sample divergent paths with probabilities proportional to the (unnormalized) reward of target problems. Extensive experiments show that, with limited training examples (e.g., 15 examples), FoR enables the discovery of diverse, creative, high-quality solutions, greatly outperforming a wide range of existing inference and training methods across six challenging reasoning tasks, including BlocksWorld (embodied reasoning), Game24 (math puzzle solving), Rubik's Cube (spatial reasoning), 1D-ARC (abstraction reasoning), GSM8k (math reasoning), and ProntoQA (logical reasoning). Code is available at https://github.com/Yu-Fangxu/FoR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08468v2">Multi-GraspLLM: A Multimodal LLM for Multi-Hand Semantic Guided Grasp Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Multi-hand semantic grasp generation aims to generate feasible and semantically appropriate grasp poses for different robotic hands based on natural language instructions. Although the task is highly valuable, due to the lack of multihand grasp datasets with fine-grained contact description between robotic hands and objects, it is still a long-standing difficult task. In this paper, we present Multi-GraspSet, the first large-scale multi-hand grasp dataset with automatically contact annotations. Based on Multi-GraspSet, we propose Multi-GraspLLM, a unified language-guided grasp generation framework, which leverages large language models (LLM) to handle variable-length sequences, generating grasp poses for diverse robotic hands in a single unified architecture. Multi-GraspLLM first aligns the encoded point cloud features and text features into a unified semantic space. It then generates grasp bin tokens that are subsequently converted into grasp pose for each robotic hand via hand-aware linear mapping. The experimental results demonstrate that our approach significantly outperforms existing methods in both real-world experiments and simulator. More information can be found on our project page https://multi-graspllm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05281v1">Similarity-Based Domain Adaptation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Unsupervised domain adaptation leverages abundant labeled data from various source domains to generalize onto unlabeled target data. Prior research has primarily focused on learning domain-invariant features across the source and target domains. However, these methods often require training a model using source domain data, which is time-consuming and can limit model usage for applications with different source data. This paper introduces a simple framework that utilizes the impressive generalization capabilities of Large Language Models (LLMs) for target data annotation without the need of source model training, followed by a novel similarity-based knowledge distillation loss. Our extensive experiments on cross-domain text classification reveal that our framework achieves impressive performance, specifically, 2.44\% accuracy improvement when compared to the SOTA method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23746v2">DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)
    </div>
    <details class="paper-abstract">
      Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05248v1">Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      The increasing adoption of large language models (LLMs) necessitates inference serving systems that can deliver both high throughput and low latency. Deploying LLMs with hundreds of billions of parameters on memory-constrained GPUs exposes significant limitations in static batching methods. Current inference serving systems often treat batch sizes as fixed hyper-parameters, hindering real-time adaptation to varying system conditions. In this paper, we propose a dynamic batching method that continuously monitors memory utilization and adheres to service-level agreements (SLAs) to enable real-time batch size configuration adjustment. The method comprises two core components: a memory-aware batch scheduler that dynamically allocates GPU resources and a latency feedback mechanism that optimizes decoding processes under SLA constraints. The numerical experiments demonstrate throughput gains of 8% to 28% and capacity improvements of 22% compared to traditional static batching methods, while maintaining full compatibility with existing inference infrastructure. These results highlight the effectiveness of dynamic batching in balancing computational efficiency and quality-of-service requirements for contemporary LLM deployment scenarios. The source code of this work is publicly available at https://github.com/KevinLee1110/dynamic-batching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05200v1">ORANSight-2.0: Foundational LLMs for O-RAN</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Despite the transformative impact of Large Language Models (LLMs) across critical domains such as healthcare, customer service, and business marketing, their integration into Open Radio Access Networks (O-RAN) remains limited. This gap is primarily due to the absence of domain-specific foundational models, with existing solutions often relying on general-purpose LLMs that fail to address the unique challenges and technical intricacies of O-RAN. To bridge this gap, we introduce ORANSight-2.0 (O-RAN Insights), a pioneering initiative aimed at developing specialized foundational LLMs tailored for O-RAN. Built on 18 LLMs spanning five open-source LLM frameworks, ORANSight-2.0 fine-tunes models ranging from 1 to 70B parameters, significantly reducing reliance on proprietary, closed-source models while enhancing performance for O-RAN. At the core of ORANSight-2.0 is RANSTRUCT, a novel Retrieval-Augmented Generation (RAG) based instruction-tuning framework that employs two LLM agents to create high-quality instruction-tuning datasets. The generated dataset is then used to fine-tune the 18 pre-trained open-source LLMs via QLoRA. To evaluate ORANSight-2.0, we introduce srsRANBench, a novel benchmark designed for code generation and codebase understanding in the context of srsRAN, a widely used 5G O-RAN stack. We also leverage ORANBench13K, an existing benchmark for assessing O-RAN-specific knowledge. Our comprehensive evaluations demonstrate that ORANSight-2.0 models outperform general-purpose and closed-source models, such as ChatGPT-4o and Gemini, by 5.421% on ORANBench and 18.465% on srsRANBench, achieving superior performance while maintaining lower computational and energy costs. We also experiment with RAG-augmented variants of ORANSight-2.0 LLMs and thoroughly evaluate their energy characteristics, demonstrating costs for training, standard inference, and RAG-augmented inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05193v1">Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance on knowledge graph question answering (KGQA) tasks by planning and interacting with knowledge graphs. However, existing methods often confuse tool utilization with knowledge reasoning, harming readability of model outputs and giving rise to hallucinatory tool invocations, which hinder the advancement of KGQA. To address this issue, we propose Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning (MemQ) to decouple LLM from tool invocation tasks using LLM-built query memory. By establishing a memory module with explicit descriptions of query statements, the proposed MemQ facilitates the KGQA process with natural language reasoning and memory-augmented query reconstruction. Meanwhile, we design an effective and readable reasoning to enhance the LLM's reasoning capability in KGQA. Experimental results that MemQ achieves state-of-the-art performance on widely used benchmarks WebQSP and CWQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05188v1">Rewarding Curse: Analyze and Mitigate Reward Modeling Issues for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 18 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) prompting demonstrates varying performance under different reasoning tasks. Previous work attempts to evaluate it but falls short in providing an in-depth analysis of patterns that influence the CoT. In this paper, we study the CoT performance from the perspective of effectiveness and faithfulness. For the former, we identify key factors that influence CoT effectiveness on performance improvement, including problem difficulty, information gain, and information flow. For the latter, we interpret the unfaithful CoT issue by conducting a joint analysis of the information interaction among the question, CoT, and answer. The result demonstrates that, when the LLM predicts answers, it can recall correct information missing in the CoT from the question, leading to the problem. Finally, we propose a novel algorithm to mitigate this issue, in which we recall extra information from the question to enhance the CoT generation and evaluate CoTs based on their information gain. Extensive experiments demonstrate that our approach enhances both the faithfulness and effectiveness of CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12643v2">LLM-based Discriminative Reasoning for Knowledge Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based on generative pre-trained Transformer have achieved remarkable performance on knowledge graph question-answering (KGQA) tasks. However, LLMs often produce ungrounded subgraph planning or reasoning results in KGQA due to the hallucinatory behavior brought by the generative paradigm. To tackle this issue, we propose READS to reformulate the KGQA process into discriminative subtasks, which simplifies the search space for each subtasks. Based on the subtasks, we design a new corresponding discriminative inference strategy to conduct the reasoning for KGQA, thereby alleviating hallucination and ungrounded reasoning issues in LLMs. Experimental results show that the proposed approach outperforms multiple strong comparison methods, along with achieving state-of-the-art performance on widely used benchmarks WebQSP and CWQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11142v3">NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05164v1">A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Evaluation methods for autonomous driving are crucial for algorithm optimization. However, due to the complexity of driving intelligence, there is currently no comprehensive evaluation method for the level of autonomous driving intelligence. In this paper, we propose an evaluation framework for driving behavior intelligence in complex traffic environments, aiming to fill this gap. We constructed a natural language evaluation dataset of human professional drivers and passengers through naturalistic driving experiments and post-driving behavior evaluation interviews. Based on this dataset, we developed an LLM-powered driving evaluation framework. The effectiveness of this framework was validated through simulated experiments in the CARLA urban traffic simulator and further corroborated by human assessment. Our research provides valuable insights for evaluating and designing more intelligent, human-like autonomous driving agents. The implementation details of the framework and detailed information about the dataset can be found at Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05142v1">RocketEval: Efficient Automated LLM Evaluation via Grading Checklist</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted by ICLR 2025: https://openreview.net/forum?id=zJjzNj6QUe
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in diverse and challenging scenarios is essential to align them with human preferences. To mitigate the prohibitive costs associated with human evaluations, utilizing a powerful LLM as a judge has emerged as a favored approach. Nevertheless, this methodology encounters several challenges, including substantial expenses, concerns regarding privacy and security, and reproducibility. In this paper, we propose a straightforward, replicable, and accurate automated evaluation method by leveraging a lightweight LLM as the judge, named RocketEval. Initially, we identify that the performance disparity between lightweight and powerful LLMs in evaluation tasks primarily stems from their ability to conduct comprehensive analyses, which is not easily enhanced through techniques such as chain-of-thought reasoning. By reframing the evaluation task as a multi-faceted Q&A using an instance-specific checklist, we demonstrate that the limited judgment accuracy of lightweight LLMs is largely attributes to high uncertainty and positional bias. To address these challenges, we introduce an automated evaluation process grounded in checklist grading, which is designed to accommodate a variety of scenarios and questions. This process encompasses the creation of checklists, the grading of these checklists by lightweight LLMs, and the reweighting of checklist items to align with the supervised annotations. Our experiments carried out on the automated evaluation benchmarks, MT-Bench and WildBench datasets, reveal that RocketEval, when using Gemma-2-2B as the judge, achieves a high correlation (0.965) with human preferences, which is comparable to GPT-4o. Moreover, RocketEval provides a cost reduction exceeding 50-fold for large-scale evaluation and comparison scenarios. Our code is available at https://github.com/Joinn99/RocketEval-ICLR .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05139v1">Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 34 pages
    </div>
    <details class="paper-abstract">
      In this technical report, we tackle the challenges of training large-scale Mixture of Experts (MoE) models, focusing on overcoming cost inefficiency and resource limitations prevalent in such systems. To address these issues, we present two differently sized MoE large language models (LLMs), namely Ling-Lite and Ling-Plus (referred to as "Bailing" in Chinese, spelled B\v{a}il\'ing in Pinyin). Ling-Lite contains 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus boasts 290 billion parameters with 28.8 billion activated parameters. Both models exhibit comparable performance to leading industry benchmarks. This report offers actionable insights to improve the efficiency and accessibility of AI development in resource-constrained settings, promoting more scalable and sustainable technologies. Specifically, to reduce training costs for large-scale MoE models, we propose innovative methods for (1) optimization of model architecture and training processes, (2) refinement of training anomaly handling, and (3) enhancement of model evaluation efficiency. Additionally, leveraging high-quality data generated from knowledge graphs, our models demonstrate superior capabilities in tool use compared to other models. Ultimately, our experimental findings demonstrate that a 300B MoE LLM can be effectively trained on lower-performance devices while achieving comparable performance to models of a similar scale, including dense and MoE models. Compared to high-performance devices, utilizing a lower-specification hardware system during the pre-training phase demonstrates significant cost savings, reducing computing costs by approximately 20%. The models can be accessed at https://huggingface.co/inclusionAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11521v2">Preempting Text Sanitization Utility in Resource-Constrained Privacy-Preserving LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Interactions with online Large Language Models raise privacy issues where providers can gather sensitive information about users and their companies from the prompts. While Differential Privacy can be applied on textual prompts through the Multidimensional Laplace Mechanism, we show that it is difficult to anticipate the utility of such sanitized prompt. Poor utility has clear monetary consequences for LLM services charging on a pay-per-use model as well as great amount of computing resources wasted. To this end, we propose an architecture to predict the utility of a given sanitized prompt before it is sent to the LLM. We experimentally show that our architecture helps prevent such resource waste for up to 12% of the prompts. We also reproduce experiments from one of the most cited paper on distance-based DP for text sanitization and show that a potential performance-driven implementation choice completely changes the output while not being explicitly defined in the paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10340v5">On the Vulnerability of LLM/VLM-Controlled Robotics</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      In this work, we highlight vulnerabilities in robotic systems integrating large language models (LLMs) and vision-language models (VLMs) due to input modality sensitivities. While LLM/VLM-controlled robots show impressive performance across various tasks, their reliability under slight input variations remains underexplored yet critical. These models are highly sensitive to instruction or perceptual input changes, which can trigger misalignment issues, leading to execution failures with severe real-world consequences. To study this issue, we analyze the misalignment-induced vulnerabilities within LLM/VLM-controlled robotic systems and present a mathematical formulation for failure modes arising from variations in input modalities. We propose empirical perturbation strategies to expose these vulnerabilities and validate their effectiveness through experiments on multiple robot manipulation tasks. Our results show that simple input perturbations reduce task execution success rates by 22.2% and 14.6% in two representative LLM/VLM-controlled robotic systems. These findings underscore the importance of input modality robustness and motivate further research to ensure the safe and reliable deployment of advanced LLM/VLM-controlled robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03704v2">A Practical Memory Injection Attack against LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Agents based on large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, that enables the injection of malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps leading to undesirable agent actions when executing the victim user's query. Specifically, we introduce a sequence of bridging steps to link the victim query to the malicious reasoning steps. During the injection of the malicious record, we propose an indication prompt to guide the agent to autonomously generate our designed bridging steps. We also propose a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing the victim query comes after. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting practical risks of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04723v2">Shifting Long-Context LLMs Research from Input to Output</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advancements in long-context Large Language Models (LLMs) have primarily concentrated on processing extended input contexts, resulting in significant strides in long-context comprehension. However, the equally critical aspect of generating long-form outputs has received comparatively less attention. This paper advocates for a paradigm shift in NLP research toward addressing the challenges of long-output generation. Tasks such as novel writing, long-term planning, and complex reasoning require models to understand extensive contexts and produce coherent, contextually rich, and logically consistent extended text. These demands highlight a critical gap in current LLM capabilities. We underscore the importance of this under-explored domain and call for focused efforts to develop foundational LLMs tailored for generating high-quality, long-form outputs, which hold immense potential for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05061v1">No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge is a framework that uses an LLM (large language model) to evaluate the quality of natural language text - typically text that is also generated by an LLM. This framework holds great promise due to its relative low-cost, ease of use, and strong correlations with human stylistic preferences. However, LLM Judges have been shown to exhibit biases that can distort their judgments. We evaluate how well LLM Judges can grade whether a given response to a conversational question is correct, an ability crucial to soundly estimating the overall response quality. To do so, we create and publicly release a human-annotated dataset with labels of correctness for 1,200 LLM responses. We source questions from a combination of existing datasets and a novel, challenging benchmark (BFF-Bench) created for this analysis. We demonstrate a strong connection between an LLM's ability to correctly answer a question and grade responses to that question. Although aggregate level statistics might imply a judge has high agreement with human annotators, it will struggle on the subset of questions it could not answer. To address this issue, we recommend a simple solution: provide the judge with a correct, human-written reference answer. We perform an in-depth analysis on how reference quality can affect the performance of an LLM Judge. We show that providing a weaker judge (e.g. Qwen 2.5 7B) with higher quality references reaches better agreement with human annotators than a stronger judge (e.g. GPT-4o) with synthetic references.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05980v1">SINdex: Semantic INconsistency Index for Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed across diverse domains, yet they are prone to generating factually incorrect outputs - commonly known as "hallucinations." Among existing mitigation strategies, uncertainty-based methods are particularly attractive due to their ease of implementation, independence from external data, and compatibility with standard LLMs. In this work, we introduce a novel and scalable uncertainty-based semantic clustering framework for automated hallucination detection. Our approach leverages sentence embeddings and hierarchical clustering alongside a newly proposed inconsistency measure, SINdex, to yield more homogeneous clusters and more accurate detection of hallucination phenomena across various LLMs. Evaluations on prominent open- and closed-book QA datasets demonstrate that our method achieves AUROC improvements of up to 9.3% over state-of-the-art techniques. Extensive ablation studies further validate the effectiveness of each component in our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v4">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve strong performances on various benchmark datasets without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05965v1">Validating LLM-as-a-Judge Systems in the Absence of Gold Labels</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm, in which a judge LLM system replaces human raters in rating the outputs of other generative AI (GenAI) systems, has come to play a critical role in scaling and standardizing GenAI evaluations. To validate judge systems, evaluators collect multiple human ratings for each item in a validation corpus, and then aggregate the ratings into a single, per-item gold label rating. High agreement rates between these gold labels and judge system ratings are then taken as a sign of good judge system performance. In many cases, however, items or rating criteria may be ambiguous, or there may be principled disagreement among human raters. In such settings, gold labels may not exist for many of the items. In this paper, we introduce a framework for LLM-as-a-judge validation in the absence of gold labels. We present a theoretical analysis drawing connections between different measures of judge system performance under different rating elicitation and aggregation schemes. We also demonstrate empirically that existing validation approaches can select judge systems that are highly suboptimal, performing as much as 34% worse than the systems selected by alternative approaches that we describe. Based on our findings, we provide concrete recommendations for developing more reliable approaches to LLM-as-a-judge validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05951v1">TPU-Gen: LLM-Driven Custom Tensor Processing Unit Generator</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 8 Pages, 9 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      The increasing complexity and scale of Deep Neural Networks (DNNs) necessitate specialized tensor accelerators, such as Tensor Processing Units (TPUs), to meet various computational and energy efficiency requirements. Nevertheless, designing optimal TPU remains challenging due to the high domain expertise level, considerable manual design time, and lack of high-quality, domain-specific datasets. This paper introduces TPU-Gen, the first Large Language Model (LLM) based framework designed to automate the exact and approximate TPU generation process, focusing on systolic array architectures. TPU-Gen is supported with a meticulously curated, comprehensive, and open-source dataset that covers a wide range of spatial array designs and approximate multiply-and-accumulate units, enabling design reuse, adaptation, and customization for different DNN workloads. The proposed framework leverages Retrieval-Augmented Generation (RAG) as an effective solution for a data-scare hardware domain in building LLMs, addressing the most intriguing issue, hallucinations. TPU-Gen transforms high-level architectural specifications into optimized low-level implementations through an effective hardware generation pipeline. Our extensive experimental evaluations demonstrate superior performance, power, and area efficiency, with an average reduction in area and power of 92\% and 96\% from the manual optimization reference values. These results set new standards for driving advancements in next-generation design automation tools powered by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19865v2">Reverse Thinking Makes LLMs Stronger Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Reverse thinking plays a crucial role in human reasoning. Humans can reason not only from a problem to a solution but also in reverse, i.e., start from the solution and reason towards the problem. This often enhances overall reasoning performance as it enables consistency checks between their forward and backward thinking. To enable Large Language Models (LLMs) to perform reverse thinking, we introduce Reverse-Enhanced Thinking (RevThink), a framework composed of data augmentation and learning objectives. In RevThink, we augment the dataset by collecting structured forward-backward reasoning from a teacher model, consisting of: (1) the original question, (2) forward reasoning, (3) backward question, and (4) backward reasoning. We then employ three objectives to train a smaller student model in a multi-task learning fashion: (a) generate forward reasoning from a question, (b) generate a backward question from a question, and (c) generate backward reasoning from the backward question. Experiments across 12 datasets covering commonsense, math, and logical reasoning show an average 13.53% improvement over the student model's zero-shot performance and a 6.84% improvement over the strongest knowledge distillation baselines. Moreover, our method demonstrates sample efficiency -- using only 10% of the correct forward reasoning from the training data, it outperforms a standard fine-tuning method trained on 10x more forward reasoning. RevThink also exhibits strong generalization to out-of-distribution held-out datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08291v3">CleanAgent: Automating Data Standardization with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Data standardization is a crucial part of the data science life cycle. While tools like Pandas offer robust functionalities, their complexity and the manual effort required for customizing code to diverse column types pose significant challenges. Although large language models (LLMs) like ChatGPT have shown promise in automating this process through natural language understanding and code generation, it still demands expert-level programming knowledge and continuous interaction for prompt refinement. To solve these challenges, our key idea is to propose a Python library with declarative, unified APIs for standardizing different column types, simplifying the LLM's code generation with concise API calls. We first propose Dataprep.Clean, a component of the Dataprep Python Library, significantly reduces the coding complexity by enabling the standardization of specific column types with a single line of code. Then, we introduce the CleanAgent framework integrating Dataprep.Clean and LLM-based agents to automate the data standardization process. With CleanAgent, data scientists only need to provide their requirements once, allowing for a hands-free process. To demonstrate the practical utility of CleanAgent, we developed a user-friendly web application, allowing attendees to interact with it using real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05856v1">This Is Your Doge, If It Please You: Exploring Deception and Robustness in Mixture of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 35 pages, 9 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Mixture of large language model (LLMs) Agents (MoA) architectures achieve state-of-the-art performance on prominent benchmarks like AlpacaEval 2.0 by leveraging the collaboration of multiple LLMs at inference time. Despite these successes, an evaluation of the safety and reliability of MoA is missing. We present the first comprehensive study of MoA's robustness against deceptive LLM agents that deliberately provide misleading responses. We examine factors like the propagation of deceptive information, model size, and information availability, and uncover critical vulnerabilities. On AlpacaEval 2.0, the popular LLaMA 3.1-70B model achieves a length-controlled Win Rate (LC WR) of 49.2% when coupled with 3-layer MoA (6 LLM agents). However, we demonstrate that introducing only a $\textit{single}$ carefully-instructed deceptive agent into the MoA can reduce performance to 37.9%, effectively nullifying all MoA gains. On QuALITY, a multiple-choice comprehension task, the impact is also severe, with accuracy plummeting by a staggering 48.5%. Inspired in part by the historical Doge of Venice voting process, designed to minimize influence and deception, we propose a range of unsupervised defense mechanisms that recover most of the lost performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05854v1">Accelerating Earth Science Discovery via Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 10 pages, 1 figure. Perspective article
    </div>
    <details class="paper-abstract">
      This Perspective explores the transformative potential of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs) in the geosciences. Users of geoscientific data repositories face challenges due to the complexity and diversity of data formats, inconsistent metadata practices, and a considerable number of unprocessed datasets. MAS possesses transformative potential for improving scientists' interaction with geoscientific data by enabling intelligent data processing, natural language interfaces, and collaborative problem-solving capabilities. We illustrate this approach with "PANGAEA GPT", a specialized MAS pipeline integrated with the diverse PANGAEA database for Earth and Environmental Science, demonstrating how MAS-driven workflows can effectively manage complex datasets and accelerate scientific discovery. We discuss how MAS can address current data challenges in geosciences, highlight advancements in other scientific fields, and propose future directions for integrating MAS into geoscientific data processing pipelines. In this Perspective, we show how MAS can fundamentally improve data accessibility, promote cross-disciplinary collaboration, and accelerate geoscientific discoveries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05846v1">Extracting and Emulsifying Cultural Explanation to Improve Multilingual Capability of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 under review, 18pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success, but their English-centric training data limits performance in non-English languages, highlighting the need for enhancements in their multilingual capabilities. While some work on multilingual prompting methods handles non-English queries by utilizing English translations or restructuring them to more closely align with LLM reasoning patterns, these works often overlook the importance of cultural context, limiting their effectiveness. To address this limitation, we propose EMCEI, a simple yet effective approach that improves LLMs' multilingual capabilities by incorporating cultural context for more accurate and appropriate responses. Specifically, EMCEI follows a two-step process that first extracts relevant cultural context from the LLM's parametric knowledge via prompting. Then, EMCEI employs an LLM-as-Judge mechanism to select the most appropriate response by balancing cultural relevance and reasoning ability. Experiments on diverse multilingual benchmarks show that EMCEI outperforms existing baselines, demonstrating its effectiveness in handling multilingual queries with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07657v1">SplitQuantV2: Enhancing Low-Bit Quantization of LLMs Without GPUs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      The quantization of large language models (LLMs) is crucial for deploying them on devices with limited computational resources. While advanced quantization algorithms offer improved performance compared to the basic linear quantization, they typically require high-end graphics processing units (GPUs), are often restricted to specific deep neural network (DNN) frameworks, and require calibration datasets. This limitation poses challenges for using such algorithms on various neural processing units (NPUs) and edge AI devices, which have diverse model formats and frameworks. In this paper, we show SplitQuantV2, an innovative algorithm designed to enhance low-bit linear quantization of LLMs, can achieve results comparable to those of advanced algorithms. SplitQuantV2 preprocesses models by splitting linear and convolution layers into functionally equivalent, quantization-friendly structures. The algorithm's platform-agnostic, concise, and efficient nature allows for implementation without the need for GPUs. Our evaluation on the Llama 3.2 1B Instruct model using the AI2's Reasoning Challenge (ARC) dataset demonstrates that SplitQuantV2 improves the accuracy of the INT4 quantization model by 11.76%p, matching the performance of the original floating-point model. Remarkably, SplitQuantV2 took only 2 minutes 6 seconds to preprocess the 1B model and perform linear INT4 quantization using only an Apple M4 CPU. SplitQuantV2 provides a practical solution for low-bit quantization on LLMs, especially when complex, computation-intensive algorithms are inaccessible due to hardware limitations or framework incompatibilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10351v4">Bias Unveiled: Investigating Social Bias in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 accepted for publication in the Association for the Advancement of Artificial Intelligence (AAAI), 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced the field of automated code generation. However, a notable research gap exists in evaluating social biases that may be present in the code produced by LLMs. To solve this issue, we propose a novel fairness framework, i.e., Solar, to assess and mitigate the social biases of LLM-generated code. Specifically, Solar can automatically generate test cases for quantitatively uncovering social biases of the auto-generated code by LLMs. To quantify the severity of social biases in generated code, we develop a dataset that covers a diverse set of social problems. We applied Solar and the crafted dataset to four state-of-the-art LLMs for code generation. Our evaluation reveals severe bias in the LLM-generated code from all the subject LLMs. Furthermore, we explore several prompting strategies for mitigating bias, including Chain-of-Thought (CoT) prompting, combining positive role-playing with CoT prompting and dialogue with Solar. Our experiments show that dialogue with Solar can effectively reduce social bias in LLM-generated code by up to 90%. Last, we make the code and data publicly available is highly extensible to evaluate new social problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00242v4">DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Update DeFT-v4, accepted by ICLR'25 (https://openreview.net/forum?id=2c7pfOqu9k). Our code is available at https://github.com/LINs-lab/DeFT
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly employed for complex tasks that process multiple generation calls in a tree structure with shared prefixes of tokens, including few-shot prompting, multi-step reasoning, speculative decoding, etc. However, existing inference systems for tree-based applications are inefficient due to improper partitioning of queries and KV cache during attention calculation. This leads to two main issues: (1) a lack of memory access (IO) reuse for KV cache of shared prefixes, and (2) poor load balancing.As a result, there is redundant KV cache IO between GPU global memory and shared memory, along with low GPU utilization. To address these challenges, we propose DeFT(Decoding with Flash Tree-Attention), a hardware-efficient attention algorithm with prefix-aware and load-balanced KV cache partitions. DeFT reduces the number of read/write operations of KV cache during attention calculation through KV-Guided Grouping, a method that avoids repeatedly loading KV cache of shared prefixes in attention computation. Additionally, we propose Flattened Tree KV Splitting, a mechanism that ensures even distribution of the KV cache across partitions with little computation redundancy, enhancing GPU utilization during attention computations. By reducing 73-99% KV cache IO and nearly 100% IO for partial results during attention calculation, DeFT achieves up to 2.23/3.59x speedup in the end-to-end/attention latency across three practical tree-based workloads compared to state-of-the-art attention algorithms. Our code is available at https://github.com/LINs-lab/DeFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05620v1">Learning LLM Preference over Intra-Dialogue Pairs: A Framework for Utterance-level Understandings</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in handling complex dialogue tasks without requiring use case-specific fine-tuning. However, analyzing live dialogues in real-time necessitates low-latency processing systems, making it impractical to deploy models with billions of parameters due to latency constraints. As a result, practitioners often prefer smaller models with millions of parameters, trained on high-quality, human-annotated datasets. Yet, curating such datasets is both time-consuming and costly. Consequently, there is a growing need to combine the scalability of LLM-generated labels with the precision of human annotations, enabling fine-tuned smaller models to achieve both higher speed and accuracy comparable to larger models. In this paper, we introduce a simple yet effective framework to address this challenge. Our approach is specifically designed for per-utterance classification problems, which encompass tasks such as intent detection, dialogue state tracking, and more. To mitigate the impact of labeling errors from LLMs -- the primary source of inaccuracies in student models -- we propose a noise-reduced preference learning loss. Experimental results demonstrate that our method significantly improves accuracy across utterance-level dialogue tasks, including sentiment detection (over $2\%$), dialogue act classification (over $1.5\%$), etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05592v1">R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Existing Large Reasoning Models (LRMs) have shown the potential of reinforcement learning (RL) to enhance the complex reasoning capabilities of Large Language Models~(LLMs). While they achieve remarkable performance on challenging tasks such as mathematics and coding, they often rely on their internal knowledge to solve problems, which can be inadequate for time-sensitive or knowledge-intensive questions, leading to inaccuracies and hallucinations. To address this, we propose \textbf{R1-Searcher}, a novel two-stage outcome-based RL approach designed to enhance the search capabilities of LLMs. This method allows LLMs to autonomously invoke external search systems to access additional knowledge during the reasoning process. Our framework relies exclusively on RL, without requiring process rewards or distillation for a cold start. % effectively generalizing to out-of-domain datasets and supporting both Base and Instruct models. Our experiments demonstrate that our method significantly outperforms previous strong RAG methods, even when compared to the closed-source GPT-4o-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17975v3">SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It)</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML 2025)
    </div>
    <details class="paper-abstract">
      Whether LLMs memorize their training data and what this means, from measuring privacy leakage to detecting copyright violations, has become a rapidly growing area of research. In the last few months, more than 10 new methods have been proposed to perform Membership Inference Attacks (MIAs) against LLMs. Contrary to traditional MIAs which rely on fixed-but randomized-records or models, these methods are mostly trained and tested on datasets collected post-hoc. Sets of members and non-members, used to evaluate the MIA, are constructed using informed guesses after the release of a model. This lack of randomization raises concerns of a distribution shift between members and non-members. In this work, we first extensively review the literature on MIAs against LLMs and show that, while most work focuses on sequence-level MIAs evaluated in post-hoc setups, a range of target models, motivations and units of interest are considered. We then quantify distribution shifts present in 6 datasets used in the literature using a model-less bag of word classifier and show that all datasets constructed post-hoc suffer from strong distribution shifts. These shifts invalidate the claims of LLMs memorizing strongly in real-world scenarios and, potentially, also the methodological contributions of the recent papers based on these datasets. Yet, all hope might not be lost. We introduce important considerations to properly evaluate MIAs against LLMs and discuss, in turn, potential ways forwards: randomized test splits, injections of randomized (unique) sequences, randomized fine-tuning, and several post-hoc control methods. While each option comes with its advantages and limitations, we believe they collectively provide solid grounds to guide MIA development and study LLM memorization. We conclude with an overview of recommended approaches to benchmark sequence-level and document-level MIAs against LLMs.
    </details>
</div>
