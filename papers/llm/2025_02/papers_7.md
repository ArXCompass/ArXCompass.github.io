# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11006v2">Effective Self-Mining of In-Context Examples for Unsupervised Machine Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Accepted at NAACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance on a wide range of natural language processing (NLP) tasks, primarily through in-context learning (ICL). In ICL, the LLM is provided with examples that represent a given task such that it learns to generate answers for test inputs. However, access to these in-context examples is not guaranteed especially for low-resource or massively multilingual tasks. In this work, we propose an unsupervised approach to mine in-context examples for machine translation (MT), enabling unsupervised MT (UMT) across different languages. Our approach begins with word-level mining to acquire word translations that are then used to perform sentence-level mining. As the quality of mined parallel pairs may not be optimal due to noise or mistakes, we introduce a filtering criterion to select the optimal in-context examples from a pool of unsupervised parallel sentences. We evaluate our approach using two multilingual LLMs on 288 directions from the FLORES-200 dataset and analyze the impact of various linguistic features on performance. Our findings demonstrate the effectiveness of our unsupervised approach in mining in-context examples for MT, leading to better or comparable translation performance as translation with regular in-context samples (extracted from human-annotated data), while also outperforming the other state-of-the-art UMT methods by an average of $7$ BLEU points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07745v3">StepTool: Enhancing Multi-Step Tool Usage in LLMs through Step-Grained Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Ongoning Work
    </div>
    <details class="paper-abstract">
      Despite powerful text generation capabilities, large language models (LLMs) still need to learn how to utilize external tools to solve complex tasks, a process known as tool learning. Existing methods primarily rely on supervised fine-tuning to enhance tool-use capabilities, treating tool learning as a text-generation task while overlooking the decision-making complexities inherent in multi-step contexts. In this work, we propose modeling tool learning as a dynamic decision-making task and introduce StepTool, a novel step-grained reinforcement learning framework that enhances the multi-step tool use capabilities of LLMs. StepTool consists of two main components: Step-grained Reward Shaping, which assigns rewards at each tool interaction based on the success of tool invocation and its contribution to the task; and Step-grained Optimization, which uses policy gradient methods to optimize the model in a multi-step manner. Experimental results demonstrate that StepTool significantly outperforms existing methods in multi-step, tool-based tasks, offering a robust solution for tool learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10674v2">Can Multimodal LLMs do Visual Temporal Understanding and Reasoning? The answer is No!</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Our dataset can be found at \url{https://huggingface.co/datasets/fazliimam/temporal-vqa}
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have achieved significant advancements in tasks like Visual Question Answering (VQA) by leveraging foundational Large Language Models (LLMs). However, their abilities in specific areas such as visual temporal understanding, which is crucial for comprehending real-world dynamics, remain underexplored. To address this, we propose a challenging evaluation benchmark named TemporalVQA, consisting of two parts: 1) Temporal Order Understanding and 2) Time-lapse Estimation. The first part requires MLLMs to determine the sequence of events by analyzing temporally consecutive video frames. The second part presents image pairs with varying time differences, framed as multiple-choice questions, asking MLLMs to estimate the time-lapse between images with options ranging from seconds to years. Our evaluations of advanced MLLMs, including models like GPT-4o and Gemini-1.5-Pro, reveal significant challenges: GPT-4o achieved only 49.1% average consistent accuracy in temporal order task and 70% in time-lapse estimation, with open-source models performing even poorly. These findings underscore the limitations of current MLLMs in visual temporal understanding and reasoning, highlighting the need for further improvements for their temporal capability. Our dataset can be found at https://huggingface.co/datasets/fazliimam/temporal-vqa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12658v1">R.R.: Unveiling LLM Training Privacy through Recollection and Ranking</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 13 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) pose significant privacy risks, potentially leaking training data due to implicit memorization. Existing privacy attacks primarily focus on membership inference attacks (MIAs) or data extraction attacks, but reconstructing specific personally identifiable information (PII) in LLM's training data remains challenging. In this paper, we propose R.R. (Recollect and Rank), a novel two-step privacy stealing attack that enables attackers to reconstruct PII entities from scrubbed training data where the PII entities have been masked. In the first stage, we introduce a prompt paradigm named recollection, which instructs the LLM to repeat a masked text but fill in masks. Then we can use PII identifiers to extract recollected PII candidates. In the second stage, we design a new criterion to score each PII candidate and rank them. Motivated by membership inference, we leverage the reference model as a calibration to our criterion. Experiments across three popular PII datasets demonstrate that the R.R. achieves better PII identical performance compared to baselines. These results highlight the vulnerability of LLMs to PII leakage even when training data has been scrubbed. We release the replicate package of R.R. at a link.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01269v5">CPRM: A LLM-based Continual Pre-training Framework for Relevance Modeling in Commercial Search</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Relevance modeling between queries and items stands as a pivotal component in commercial search engines, directly affecting the user experience. Given the remarkable achievements of large language models (LLMs) in various natural language processing (NLP) tasks, LLM-based relevance modeling is gradually being adopted within industrial search systems. Nevertheless, foundational LLMs lack domain-specific knowledge and do not fully exploit the potential of in-context learning. Furthermore, structured item text remains underutilized, and there is a shortage in the supply of corresponding queries and background knowledge. We thereby propose CPRM (Continual Pre-training for Relevance Modeling), a framework designed for the continual pre-training of LLMs to address these issues. Our CPRM framework includes three modules: 1) employing both queries and multi-field item to jointly pre-train for enhancing domain knowledge, 2) applying in-context pre-training, a novel approach where LLMs are pre-trained on a sequence of related queries or items, and 3) conducting reading comprehension on items to produce associated domain knowledge and background information (e.g., generating summaries and corresponding queries) to further strengthen LLMs. Results on offline experiments and online A/B testing demonstrate that our model achieves convincing performance compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12499v2">With a Grain of SALT: Are LLMs Fair Across Social Dimensions?</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      This paper presents a systematic analysis of biases in open-source Large Language Models (LLMs), across gender, religion, and race. Our study evaluates bias in smaller-scale Llama and Gemma models using the SALT ($\textbf{S}$ocial $\textbf{A}$ppropriateness in $\textbf{L}$LM-Generated $\textbf{T}$ext) dataset, which incorporates five distinct bias triggers: General Debate, Positioned Debate, Career Advice, Problem Solving, and CV Generation. To quantify bias, we measure win rates in General Debate and the assignment of negative roles in Positioned Debate. For real-world use cases, such as Career Advice, Problem Solving, and CV Generation, we anonymize the outputs to remove explicit demographic identifiers and use DeepSeek-R1 as an automated evaluator. We also address inherent biases in LLM-based evaluation, including evaluation bias, positional bias, and length bias, and validate our results through human evaluations. Our findings reveal consistent polarization across models, with certain demographic groups receiving systematically favorable or unfavorable treatment. By introducing SALT, we provide a comprehensive benchmark for bias analysis and underscore the need for robust bias mitigation strategies in the development of equitable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10858v2">Is Depth All You Need? An Exploration of Iterative Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 22 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Deep iterative chain-of-thought (CoT) reasoning enables LLMs to tackle complex tasks by progressively activating relevant pre-trained knowledge. However, it faces challenges in ensuring continual improvement and determining a stopping criterion. In this paper, we investigate whether the relevant knowledge that contributes directly to solving the given question can be activated from the initial reasoning path, thus circumventing the need for iterative refinement. Our experiments reveal that increasing the diversity of initial reasoning paths can achieve comparable or superior performance, a concept we term \textit{breadth reasoning}. However, existing breadth reasoning approaches, such as self-consistency, offer limited diversity. To address this limitation, we propose a simple yet effective method that enhances reasoning breadth by integrating contextual exploration with reduced sampling randomness. Extensive experiments demonstrate that our approach significantly outperforms deep iterative reasoning. Our code is provided in https://github.com/zongqianwu/breadth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12598v1">Bring Your Own Knowledge: A Survey of Methods for LLM Knowledge Expansion</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Adapting large language models (LLMs) to new and diverse knowledge is essential for their lasting effectiveness in real-world applications. This survey provides an overview of state-of-the-art methods for expanding the knowledge of LLMs, focusing on integrating various knowledge types, including factual information, domain expertise, language proficiency, and user preferences. We explore techniques, such as continual learning, model editing, and retrieval-based explicit adaptation, while discussing challenges like knowledge consistency and scalability. Designed as a guide for researchers and practitioners, this survey sheds light on opportunities for advancing LLMs as adaptable and robust knowledge systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09077v3">CuriousLLM: Elevating Multi-Document Question Answering with LLM-Enhanced Knowledge Graph Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Accepted for publication in NAACL 2025. The official version will be available in the ACL Anthology
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant success in open-domain question answering. However, they continue to face challenges such as hallucinations and knowledge cutoffs. These issues can be mitigated through in-context learning by providing LLMs with relevant context before generating answers. Recent literature proposes Knowledge Graph Prompting (KGP) which integrates knowledge graphs with an LLM-based traversal agent to substantially enhance document retrieval quality. However, KGP requires costly fine-tuning with large datasets and remains prone to hallucination. In this paper, we propose CuriousLLM, an enhancement that integrates a curiosity-driven reasoning mechanism into an LLM agent. This mechanism enables the agent to generate relevant follow-up questions, thereby guiding the information retrieval process more efficiently. Central to our approach is the development of the new Follow-upQA dataset, which includes questions and supporting evidence as input, with follow-up questions serving as ground truths. These follow-up questions either inquire about what is still missing to fully answer the user's query or use special tokens to signify that the retrieved evidence is sufficient. Our experiments show that CuriousLLM significantly boosts LLM performance in multi-document question answering (MD-QA), circumventing the substantial computational costs and latency from the original KGP framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12018v1">Atom of Thoughts for Markov LLM Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through training-time scaling, and test-time scaling further enhances their capabilities by conducting effective reasoning during inference. However, as the scale of reasoning increases, existing test-time scaling methods suffer from accumulated historical information, which not only wastes computational resources but also interferes with effective reasoning. To address this issue, we observe that complex reasoning progress is often achieved by solving a sequence of independent subquestions, each being self-contained and verifiable. These subquestions are essentially atomic questions, relying primarily on their current state rather than accumulated history, similar to the memoryless transitions in a Markov process. Based on this observation, we propose Atom of Thoughts (AoT), where each state transition in the reasoning process consists of decomposing the current question into a dependency-based directed acyclic graph and contracting its subquestions, forming a new atomic question state. This iterative decomposition-contraction process continues until reaching directly solvable atomic questions, naturally realizing Markov transitions between question states. Furthermore, these atomic questions can be seamlessly integrated into existing test-time scaling methods, enabling AoT to serve as a plug-in enhancement for improving reasoning capabilities. Experiments across six benchmarks demonstrate the effectiveness of AoT both as a standalone framework and a plug-in enhancement. Notably, on HotpotQA, when applied to gpt-4o-mini, AoT achieves an 80.6% F1 score, surpassing o3-mini by 3.4% and DeepSeek-R1 by 10.6%. The code will be available at https://github.com/qixucen/atom.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11995v1">Presumed Cultural Identity: How Names Shape LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 23 Pages, 13 Figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Names are deeply tied to human identity. They can serve as markers of individuality, cultural heritage, and personal history. However, using names as a core indicator of identity can lead to over-simplification of complex identities. When interacting with LLMs, user names are an important point of information for personalisation. Names can enter chatbot conversations through direct user input (requested by chatbots), as part of task contexts such as CV reviews, or as built-in memory features that store user information for personalisation. We study biases associated with names by measuring cultural presumptions in the responses generated by LLMs when presented with common suggestion-seeking queries, which might involve making assumptions about the user. Our analyses demonstrate strong assumptions about cultural identity associated with names present in LLM generations across multiple cultures. Our work has implications for designing more nuanced personalisation systems that avoid reinforcing stereotypes while maintaining meaningful customisation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14483v2">Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Deciding which large language model (LLM) to use is a complex challenge. Pairwise ranking has emerged as a new method for evaluating human preferences for LLMs. This approach entails humans evaluating pairs of model outputs based on a predefined criterion. By collecting these comparisons, a ranking can be constructed using methods such as Elo. However, applying these algorithms as constructed in the context of LLM evaluation introduces several challenges. In this paper, we explore the effectiveness of ranking systems for head-to-head comparisons of LLMs. We formally define a set of fundamental principles for effective ranking and conduct a series of extensive evaluations on the robustness of several ranking algorithms in the context of LLMs. Our analysis uncovers key insights into the factors that affect ranking accuracy and efficiency, offering guidelines for selecting the most appropriate methods based on specific evaluation contexts and resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18547v4">Token-Budget-Aware LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Reasoning is critical for large language models (LLMs) to excel in a wide range of tasks. While methods like Chain-of-Thought (CoT) reasoning enhance LLM performance by decomposing problems into intermediate steps, they also incur significant overhead in token usage, leading to increased costs. We find that the reasoning process of current LLMs is unnecessarily lengthy and it can be compressed by including a reasonable token budget in the prompt, but the choice of token budget plays a crucial role in the actual compression effectiveness. We then propose a token-budget-aware LLM reasoning framework, which dynamically estimates token budgets for different problems based on reasoning complexity and uses the estimated token budgets to guide the reasoning process. Experiments show that our method effectively reduces token costs in CoT reasoning with only a slight performance reduction, offering a practical solution to balance efficiency and accuracy in LLM reasoning. Code: https://github.com/GeniusHTX/TALE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11919v1">From Text to Trust: Empowering AI-assisted Decision Making with Adaptive LLM-powered Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 CHI 2025
    </div>
    <details class="paper-abstract">
      AI-assisted decision making becomes increasingly prevalent, yet individuals often fail to utilize AI-based decision aids appropriately especially when the AI explanations are absent, potentially as they do not %understand reflect on AI's decision recommendations critically. Large language models (LLMs), with their exceptional conversational and analytical capabilities, present great opportunities to enhance AI-assisted decision making in the absence of AI explanations by providing natural-language-based analysis of AI's decision recommendation, e.g., how each feature of a decision making task might contribute to the AI recommendation. In this paper, via a randomized experiment, we first show that presenting LLM-powered analysis of each task feature, either sequentially or concurrently, does not significantly improve people's AI-assisted decision performance. To enable decision makers to better leverage LLM-powered analysis, we then propose an algorithmic framework to characterize the effects of LLM-powered analysis on human decisions and dynamically decide which analysis to present. Our evaluation with human subjects shows that this approach effectively improves decision makers' appropriate reliance on AI in AI-assisted decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13172v1">Unveiling Privacy Risks in LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents have become increasingly prevalent across various real-world applications. They enhance decision-making by storing private user-agent interactions in the memory module for demonstrations, introducing new privacy risks for LLM agents. In this work, we systematically investigate the vulnerability of LLM agents to our proposed Memory EXTRaction Attack (MEXTRA) under a black-box setting. To extract private information from memory, we propose an effective attacking prompt design and an automated prompt generation method based on different levels of knowledge about the LLM agent. Experiments on two representative agents demonstrate the effectiveness of MEXTRA. Moreover, we explore key factors influencing memory leakage from both the agent's and the attacker's perspectives. Our findings highlight the urgent need for effective memory safeguards in LLM agent design and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11910v1">Adversarial Alignment for LLMs Requires Simpler, Reproducible, and More Measurable Objectives</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Misaligned research objectives have considerably hindered progress in adversarial robustness research over the past decade. For instance, an extensive focus on optimizing target metrics, while neglecting rigorous standardized evaluation, has led researchers to pursue ad-hoc heuristic defenses that were seemingly effective. Yet, most of these were exposed as flawed by subsequent evaluations, ultimately contributing little measurable progress to the field. In this position paper, we illustrate that current research on the robustness of large language models (LLMs) risks repeating past patterns with potentially worsened real-world implications. To address this, we argue that realigned objectives are necessary for meaningful progress in adversarial alignment. To this end, we build on established cybersecurity taxonomy to formally define differences between past and emerging threat models that apply to LLMs. Using this framework, we illustrate that progress requires disentangling adversarial alignment into addressable sub-problems and returning to core academic principles, such as measureability, reproducibility, and comparability. Although the field presents significant challenges, the fresh start on adversarial robustness offers the unique opportunity to build on past experience while avoiding previous mistakes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11880v1">Bitnet.cpp: Efficient Edge Inference for Ternary LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 18 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The advent of 1-bit large language models (LLMs), led by BitNet b1.58, has spurred interest in ternary LLMs. Despite this, research and practical applications focusing on efficient edge inference for ternary LLMs remain scarce. To bridge this gap, we introduce Bitnet.cpp, an inference system optimized for BitNet b1.58 and ternary LLMs. Given that mixed-precision matrix multiplication (mpGEMM) constitutes the bulk of inference time in ternary LLMs, Bitnet.cpp incorporates a novel mpGEMM library to facilitate sub-2-bits-per-weight, efficient and lossless inference. The library features two core solutions: Ternary Lookup Table (TL), which addresses spatial inefficiencies of previous bit-wise methods, and Int2 with a Scale (I2_S), which ensures lossless edge inference, both enabling high-speed inference. Our experiments show that Bitnet.cpp achieves up to a 6.25x increase in speed over full-precision baselines and up to 2.32x over low-bit baselines, setting new benchmarks in the field. Additionally, we expand TL to element-wise lookup table (ELUT) for low-bit LLMs in the appendix, presenting both theoretical and empirical evidence of its considerable potential. Bitnet.cpp is publicly available at https://github.com/microsoft/BitNet/tree/paper , offering a sophisticated solution for the efficient and practical deployment of edge LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11877v1">JoLT: Joint Probabilistic Predictions on Tabular Data Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      We introduce a simple method for probabilistic predictions on tabular data based on Large Language Models (LLMs) called JoLT (Joint LLM Process for Tabular data). JoLT uses the in-context learning capabilities of LLMs to define joint distributions over tabular data conditioned on user-specified side information about the problem, exploiting the vast repository of latent problem-relevant knowledge encoded in LLMs. JoLT defines joint distributions for multiple target variables with potentially heterogeneous data types without any data conversion, data preprocessing, special handling of missing data, or model training, making it accessible and efficient for practitioners. Our experiments show that JoLT outperforms competitive methods on low-shot single-target and multi-target tabular classification and regression tasks. Furthermore, we show that JoLT can automatically handle missing data and perform data imputation by leveraging textual side information. We argue that due to its simplicity and generality, JoLT is an effective approach for a wide variety of real prediction problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11863v1">FedEAT: A Robustness Optimization Framework for Federated LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11856v1">LLMs as a synthesis between symbolic and continuous approaches to language</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Since the middle of the 20th century, a fierce battle is being fought between symbolic and continuous approaches to language and cognition. The success of deep learning models, and LLMs in particular, has been alternatively taken as showing that the continuous camp has won, or dismissed as an irrelevant engineering development. However, in this position paper I argue that deep learning models for language actually represent a synthesis between the two traditions. This is because 1) deep learning architectures allow for both continuous/distributed and symbolic/discrete-like representations and computations; 2) models trained on language make use this flexibility. In particular, I review recent research in mechanistic interpretability that showcases how a substantial part of morphosyntactic knowledge is encoded in a near-discrete fashion in LLMs. This line of research suggests that different behaviors arise in an emergent fashion, and models flexibly alternate between the two modes (and everything in between) as needed. This is possibly one of the main reasons for their wild success; and it is also what makes them particularly interesting for the study of language and cognition. Is it time for peace?
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11844v1">BaxBench: Can LLMs Generate Correct and Secure Backends?</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      The automatic generation of programs has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 60% on code correctness; (ii) on average, we could successfully execute security exploits on more than half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11843v1">Can LLM Agents Maintain a Persona in Discourse?</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used as conversational agents, exploiting their capabilities in various sectors such as education, law, medicine, and more. However, LLMs are often subjected to context-shifting behaviour, resulting in a lack of consistent and interpretable personality-aligned interactions. Adherence to psychological traits lacks comprehensive analysis, especially in the case of dyadic (pairwise) conversations. We examine this challenge from two viewpoints, initially using two conversation agents to generate a discourse on a certain topic with an assigned personality from the OCEAN framework (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) as High/Low for each trait. This is followed by using multiple judge agents to infer the original traits assigned to explore prediction consistency, inter-model agreement, and alignment with the assigned personality. Our findings indicate that while LLMs can be guided toward personality-driven dialogue, their ability to maintain personality traits varies significantly depending on the combination of models and discourse settings. These inconsistencies emphasise the challenges in achieving stable and interpretable personality-aligned interactions in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14838v2">DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Efficient KV cache management in LLMs is crucial for long-context tasks like RAG and summarization. Existing KV cache compression methods enforce a fixed pattern, neglecting task-specific characteristics and reducing the retention of essential information. However, we observe distinct activation patterns across layers in various tasks, highlighting the need for adaptive strategies tailored to each task's unique demands. Based on this insight, we propose DynamicKV, a method that dynamically optimizes token retention by adjusting the number of tokens retained at each layer to adapt to the specific task. DynamicKV establishes global and per-layer maximum KV cache budgets, temporarily retaining the maximum budget for the current layer, and periodically updating the KV cache sizes of all preceding layers during inference. Our method retains only 1.7% of the KV cache size while achieving ~85% of the Full KV cache performance on LongBench. Notably, even under extreme compression (0.9%), DynamicKV surpasses state-of-the-art (SOTA) methods by 11% in the Needle-in-a-Haystack test using Mistral-7B-Instruct-v0.2. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11830v1">Text Classification in the LLM Era - Where do we stand?</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Large Language Models revolutionized NLP and showed dramatic performance improvements across several tasks. In this paper, we investigated the role of such language models in text classification and how they compare with other approaches relying on smaller pre-trained language models. Considering 32 datasets spanning 8 languages, we compared zero-shot classification, few-shot fine-tuning and synthetic data based classifiers with classifiers built using the complete human labeled dataset. Our results show that zero-shot approaches do well for sentiment classification, but are outperformed by other approaches for the rest of the tasks, and synthetic data sourced from multiple LLMs can build better classifiers than zero-shot open LLMs. We also see wide performance disparities across languages in all the classification scenarios. We expect that these findings would guide practitioners working on developing text classification systems across languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11812v1">Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Fine-tuning significantly improves the performance of Large Language Models (LLMs), yet its underlying mechanisms remain poorly understood. This paper aims to provide an in-depth interpretation of the fine-tuning process through circuit analysis, a popular tool in Mechanistic Interpretability (MI). Unlike previous studies \cite{prakash2024finetuningenhancesexistingmechanisms,chhabra2024neuroplasticity} that focus on tasks where pre-trained models already perform well, we develop a set of mathematical tasks where fine-tuning yields substantial performance gains, which are closer to the practical setting. In our experiments, we identify circuits at various checkpoints during fine-tuning and examine the interplay between circuit analysis, fine-tuning methods, and task complexities. First, we find that while circuits maintain high node similarity before and after fine-tuning, their edges undergo significant changes, which is in contrast to the previous work \cite{prakash2024finetuningenhancesexistingmechanisms,chhabra2024neuroplasticity} that show circuits only add some additional components after fine-tuning. Based on these observations, we develop a circuit-aware Low-Rank Adaptation (LoRA) method, which assigns ranks to layers based on edge changes in the circuits. Experimental results demonstrate that our circuit-based LoRA algorithm achieves an average performance improvement of 2.46\% over standard LoRA with similar parameter sizes. Furthermore, we explore how combining circuits from subtasks can enhance fine-tuning in compositional tasks, providing new insights into the design of such tasks and deepening the understanding of circuit dynamics and fine-tuning mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10711v3">How Should We Build A Benchmark? Revisiting 274 Code-Related Benchmarks For LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Various benchmarks have been proposed to assess the performance of large language models (LLMs) in different coding scenarios. We refer to them as code-related benchmarks. However, there are no systematic guidelines by which such a benchmark should be developed to ensure its quality, reliability, and reproducibility. We propose How2Bench, which is comprised of a 55-criteria checklist as a set of guidelines to govern the development of code-related benchmarks comprehensively. Using HOW2BENCH, we profiled 274 benchmarks released within the past decade and found concerning issues. Nearly 70% of the benchmarks did not take measures for data quality assurance; over 10% did not even open source or only partially open source. Many highly cited benchmarks have loopholes, including duplicated samples, incorrect reference codes/tests/prompts, and unremoved sensitive/confidential information. Finally, we conducted a human study involving 49 participants, which revealed significant gaps in awareness of the importance of data quality, reproducibility, and transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19318v3">TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 https://tablellm.github.io/
    </div>
    <details class="paper-abstract">
      We introduce TableLLM, a robust large language model (LLM) with 8 billion parameters, purpose-built for proficiently handling tabular data manipulation tasks, whether they are embedded within documents or spreadsheets, catering to real-world office scenarios. We propose a distant supervision method for training, which comprises a reasoning process extension strategy, aiding in training LLMs to understand reasoning patterns more effectively as well as a cross-way validation strategy, ensuring the quality of the automatically generated data. To evaluate the performance of TableLLM, we have crafted benchmarks tailored to address both document and spreadsheet formats as well as constructed a well-organized evaluation pipeline capable of handling both scenarios. Thorough evaluations underscore the advantages of TableLLM when compared to various existing general-purpose and tabular data-focused LLMs. We have publicly released the model checkpoint, source code, benchmarks, and a web application for user interaction. Our codes and data are publicly available at https://github.com/TableLLM/TableLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09056v2">Adapting Language-Specific LLMs to a Reasoning Model in One Day via Model Merging - An Open Recipe</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      This paper investigates data selection and model merging methodologies aimed at incorporating advanced reasoning capabilities such as those of DeepSeek R1 into language-specific large language models (LLMs), with a particular focus on the Thai LLM. Our goal is to enhance the reasoning capabilities of language-specific LLMs while maintaining their target language abilities. DeepSeek R1 excels in reasoning but primarily benefits high-resource languages such as English and Chinese. However, low-resource languages remain underserved due to the dominance of English-centric training data and model optimizations, which limit performance in these languages. This limitation results in unreliable code-switching and diminished effectiveness on tasks in low-resource languages. Meanwhile, local and regional LLM initiatives have attempted to bridge this gap by developing language-specific LLMs that focus on improving local linguistic fidelity. We demonstrate that, with only publicly available datasets and a computational budget of $120, it is possible to enhance the reasoning capabilities of language-specific LLMs to match the level of DeepSeek R1, without compromising their performance on target language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13669v2">How to Alleviate Catastrophic Forgetting in LLMs Finetuning? Hierarchical Layer-Wise and Element-Wise Regularization</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong general language capabilities. However, fine-tuning these models on domain-specific tasks often leads to catastrophic forgetting, where the model overwrites or loses essential knowledge acquired during pretraining. This phenomenon significantly limits the broader applicability of LLMs. To address this challenge, we propose a novel approach to compute the element-wise importance of model parameters crucial for preserving general knowledge during fine-tuning. Our method utilizes a dual-objective optimization strategy: (1) regularization loss based on element-wise parameter importance, which constrains the updates to parameters crucial for general knowledge; (2) cross-entropy loss to adapt to domain-specific tasks. Additionally, we introduce layer-wise coefficients to account for the varying contributions of different layers, dynamically balancing the dual-objective optimization. Extensive experiments on scientific, medical, and physical tasks using GPT-J and LLaMA-3 demonstrate that our approach mitigates catastrophic forgetting while enhancing model adaptability. Compared to previous methods, our solution is approximately 20 times faster and requires only 10-15% of the storage, highlighting the practical efficiency. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16207v2">From Informal to Formal -- Incorporating and Evaluating LLMs on Natural Language Requirements to Verifiable Formal Proofs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      The research in AI-based formal mathematical reasoning has shown an unstop- pable growth trend. These studies have excelled in mathematical competitions like IMO and have made significant progress. This paper focuses on formal verification, an immediate application scenario of formal reasoning, and breaks it down into sub-tasks. We constructed 18k high-quality instruction-response pairs across five formal specification languages (Coq, Lean4, Dafny, ACSL, and TLA+) by distilling gpt-4o and evaluated against ten open-sourced LLMs, including recent popular DeepSeek-R1. We also fine-tuned several 7~8B small models to achieve comparable performance with Deepseek-R1-671B. Interestingly, we observed that fine-tuning with formal data also enhances mathematics, reasoning, and coding capabilities. Fine-tuned models are released at https: //huggingface.co/fm-universe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11767v1">From Selection to Generation: A Survey of LLM-based Active Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Active Learning (AL) has been a powerful paradigm for improving model efficiency and performance by selecting the most informative data points for labeling and training. In recent active learning frameworks, Large Language Models (LLMs) have been employed not only for selection but also for generating entirely new data instances and providing more cost-effective annotations. Motivated by the increasing importance of high-quality data and efficient model training in the era of LLMs, we present a comprehensive survey on LLM-based Active Learning. We introduce an intuitive taxonomy that categorizes these techniques and discuss the transformative roles LLMs can play in the active learning loop. We further examine the impact of AL on LLM learning paradigms and its applications across various domains. Finally, we identify open challenges and propose future research directions. This survey aims to serve as an up-to-date resource for researchers and practitioners seeking to gain an intuitive understanding of LLM-based AL techniques and deploy them to new applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11751v1">Language Models Can See Better: Visual Contrastive Decoding For LLM Multimodal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) excel in reasoning and generation for language tasks, they are not specifically designed for multimodal challenges. Training Multimodal Large Language Models (MLLMs), however, is resource-intensive and constrained by various training limitations. In this paper, we propose the Modular-based Visual Contrastive Decoding (MVCD) framework to move this obstacle. Our framework leverages LLMs' In-Context Learning (ICL) capability and the proposed visual contrastive-example decoding (CED), specifically tailored for this framework, without requiring any additional training. By converting visual signals into text and focusing on contrastive output distributions during decoding, we can highlight the new information introduced by contextual examples, explore their connections, and avoid over-reliance on prior encoded knowledge. MVCD enhances LLMs' visual perception to make it see and reason over the input visuals. To demonstrate MVCD's effectiveness, we conduct experiments with four LLMs across five question answering datasets. Our results not only show consistent improvement in model accuracy but well explain the effective components inside our decoding strategy. Our code will be available at https://github.com/Pbhgit/MVCD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11723v1">Energy-Conscious LLM Decoding: Impact of Text Generation Strategies on GPU Energy Consumption</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Decoding strategies significantly influence the quality and diversity of the generated texts in large language models (LLMs), yet their impact on computational resource consumption, particularly GPU energy usage, is insufficiently studied. This paper investigates the relationship between text generation decoding methods and energy efficiency, focusing on the trade-off between generation quality and GPU energy consumption across diverse tasks and decoding configurations. By benchmarking multiple strategies across different text generation tasks, such as Translation, Code Summarization, and Math Problem Solving, we reveal how selecting appropriate decoding techniques with their tuned hyperparameters affects text quality and has measurable implications for resource utilization, emphasizing the need for balanced optimization. To the best of our knowledge, this study is among the first to explore decoding strategies in LLMs through the lens of energy consumption, offering actionable insights for designing resource-aware applications that maintain high-quality text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13126v2">Preference Curriculum: LLMs Should Always Be Pretrained on Their Preferred Data</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 18 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) generally utilize a consistent data distribution throughout the pretraining process. However, as the model's capability improves, it is intuitive that its data preferences dynamically change, indicating the need for pretraining with different data at various training stages. To achieve it, we propose the Perplexity Difference (PD) based Preference Curriculum learning (PDPC) framework, which always perceives and uses the data preferred by LLMs to train and boost them. First, we introduce the PD metric to quantify the difference in how challenging a sample is for weak versus strong models. Samples with high PD are more challenging for weak models to learn and are more suitable to be arranged in the later stage of pretraining. Second, we propose the preference function to approximate and predict the data preference of the LLM at any training step, so as to complete the arrangement of the dataset offline and ensure continuous training without interruption. Experimental results on 1.3B and 3B models demonstrate that PDPC significantly surpasses baselines. Notably, the 3B model trained on 1T tokens achieves an increased average accuracy of over 8.1% across MMLU and CMMLU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11705v1">LLM Agents Making Agent Tools</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Tool use has turned large language models (LLMs) into powerful agents that can perform complex multi-step tasks by dynamically utilising external software components. However, these tools must be implemented in advance by human developers, hindering the applicability of LLM agents in domains which demand large numbers of highly specialised tools, like in life sciences and medicine. Motivated by the growing trend of scientific studies accompanied by public code repositories, we propose ToolMaker, a novel agentic framework that autonomously transforms papers with code into LLM-compatible tools. Given a short task description and a repository URL, ToolMaker autonomously installs required dependencies and generates code to perform the task, using a closed-loop self-correction mechanism to iteratively diagnose and rectify errors. To evaluate our approach, we introduce a benchmark comprising 15 diverse and complex computational tasks spanning both medical and non-medical domains with over 100 unit tests to objectively assess tool correctness and robustness. ToolMaker correctly implements 80% of the tasks, substantially outperforming current state-of-the-art software engineering agents. ToolMaker therefore is a step towards fully autonomous agent-based scientific workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11689v1">Improve LLM-as-a-Judge Ability as a General Ability</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge leverages the generative and reasoning capabilities of large language models (LLMs) to evaluate LLM responses across diverse scenarios, providing accurate preference signals. This approach plays a vital role in aligning LLMs with human values, ensuring ethical and reliable AI outputs that align with societal norms. Recent studies have raised many methods to train LLM as generative judges, but most of them are data consuming or lack accuracy, and only focus on LLM's judge ability. In this work, we regard judge ability as a general ability of LLM and implement a two-stage training approach, comprising supervised fine-tuning (SFT) warm-up and direct preference optimization (DPO) enhancement, to achieve judge style adaptation and improve judgment accuracy. Additionally, we introduce an efficient data synthesis method to generate judgmental content. Experimental results demonstrate that our approach, utilizing only about 2% to 40% of the data required by other methods, achieves SOTA performance on RewardBench. Furthermore, our training method enhances the general capabilities of the model by constructing complicated judge task, and the judge signals provided by our model have significantly enhanced the downstream DPO training performance of our internal models in our test to optimize policy model with Judge Model. We also open-source our model weights and training data to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11677v1">Towards Fully Exploiting LLM Internal States to Enhance Knowledge Boundary Perception</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive performance across diverse tasks but often struggle to accurately gauge their knowledge boundaries, leading to confident yet incorrect responses. This paper explores leveraging LLMs' internal states to enhance their perception of knowledge boundaries from efficiency and risk perspectives. We investigate whether LLMs can estimate their confidence using internal states before response generation, potentially saving computational resources. Our experiments on datasets like Natural Questions, HotpotQA, and MMLU reveal that LLMs demonstrate significant pre-generation perception, which is further refined post-generation, with perception gaps remaining stable across varying conditions. To mitigate risks in critical domains, we introduce Consistency-based Confidence Calibration ($C^3$), which assesses confidence consistency through question reformulation. $C^3$ significantly improves LLMs' ability to recognize their knowledge gaps, enhancing the unknown perception rate by 5.6\% on NQ and 4.9\% on HotpotQA. Our findings suggest that pre-generation confidence estimation can optimize efficiency, while $C^3$ effectively controls output risks, advancing the reliability of LLMs in practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02243v2">Language Writ Large: LLMs, ChatGPT, Grounding, Meaning and Understanding</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 54 pages, 29 references
    </div>
    <details class="paper-abstract">
      Apart from what (little) OpenAI may be concealing from us, we all know (roughly) how ChatGPT works (its huge text database, its statistics, its vector representations, and their huge number of parameters, its next-word training, and so on). But none of us can say (hand on heart) that we are not surprised by what ChatGPT has proved to be able to do with these resources. This has even driven some of us to conclude that ChatGPT actually understands. It is not true that it understands. But it is also not true that we understand how it can do what it can do. I will suggest some hunches about benign biases: convergent constraints that emerge at LLM scale that may be helping ChatGPT do so much better than we would have expected. These biases are inherent in the nature of language itself, at LLM scale, and they are closely linked to what it is that ChatGPT lacks, which is direct sensorimotor grounding to connect its words to their referents and its propositions to their meanings. These convergent biases are related to (1) the parasitism of indirect verbal grounding on direct sensorimotor grounding, (2) the circularity of verbal definition, (3) the mirroring of language production and comprehension, (4) iconicity in propositions at LLM scale, (5) computational counterparts of human categorical perception in category learning by neural nets, and perhaps also (6) a conjecture by Chomsky about the laws of thought. The exposition will be in the form of a dialogue with ChatGPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11649v1">Competing LLM Agents in a Non-Cooperative Game of Opinion Polarisation</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      We introduce a novel non-cooperative game to analyse opinion formation and resistance, incorporating principles from social psychology such as confirmation bias, resource constraints, and influence penalties. Our simulation features Large Language Model (LLM) agents competing to influence a population, with penalties imposed for generating messages that propagate or counter misinformation. This framework integrates resource optimisation into the agents' decision-making process. Our findings demonstrate that while higher confirmation bias strengthens opinion alignment within groups, it also exacerbates overall polarisation. Conversely, lower confirmation bias leads to fragmented opinions and limited shifts in individual beliefs. Investing heavily in a high-resource debunking strategy can initially align the population with the debunking agent, but risks rapid resource depletion and diminished long-term influence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11620v1">Assessing Correctness in LLM-Based Code Generation via Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 18 pages and 3 References Pages
    </div>
    <details class="paper-abstract">
      In this work, we explore uncertainty estimation as a proxy for correctness in LLM-generated code. To this end, we adapt two state-of-the-art techniques from natural language generation -- one based on entropy and another on mutual information -- to the domain of code generation. Given the distinct semantic properties of code, we introduce modifications, including a semantic equivalence check based on symbolic execution. Our findings indicate a correlation between the uncertainty computed through these techniques and correctness, highlighting the potential of uncertainty estimation for quality assessment. Additionally, we propose a simplified version of the entropy-based method that assumes a uniform distribution over the LLM's responses, demonstrating comparable effectiveness. Using these techniques, we develop an abstention policy that prevents the model from making predictions when uncertainty is high, reducing incorrect outputs to near zero. Our evaluation on the LiveCodeBench shows that our approach significantly outperforms a baseline relying solely on LLM-reported log-probabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11598v1">Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 22 pages, 12 figures, 13 tables
    </div>
    <details class="paper-abstract">
      The radioactive nature of Large Language Model (LLM) watermarking enables the detection of watermarks inherited by student models when trained on the outputs of watermarked teacher models, making it a promising tool for preventing unauthorized knowledge distillation. However, the robustness of watermark radioactivity against adversarial actors remains largely unexplored. In this paper, we investigate whether student models can acquire the capabilities of teacher models through knowledge distillation while avoiding watermark inheritance. We propose two categories of watermark removal approaches: pre-distillation removal through untargeted and targeted training data paraphrasing (UP and TP), and post-distillation removal through inference-time watermark neutralization (WN). Extensive experiments across multiple model pairs, watermarking schemes and hyper-parameter settings demonstrate that both TP and WN thoroughly eliminate inherited watermarks, with WN achieving this while maintaining knowledge transfer efficiency and low computational overhead. Given the ongoing deployment of watermarking techniques in production LLMs, these findings emphasize the urgent need for more robust defense strategies. Our code is available at https://github.com/THU-BPM/Watermark-Radioactivity-Attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11596v1">LLM Embeddings for Deep Learning on Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Tabular deep-learning methods require embedding numerical and categorical input features into high-dimensional spaces before processing them. Existing methods deal with this heterogeneous nature of tabular data by employing separate type-specific encoding approaches. This limits the cross-table transfer potential and the exploitation of pre-trained knowledge. We propose a novel approach that first transforms tabular data into text, and then leverages pre-trained representations from LLMs to encode this data, resulting in a plug-and-play solution to improv ing deep-learning tabular methods. We demonstrate that our approach improves accuracy over competitive models, such as MLP, ResNet and FT-Transformer, by validating on seven classification datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11578v1">Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant strides in natural language generation but often face challenges in tasks requiring precise calculations and structural analysis. This paper investigates the performance of state-of-the-art LLMs on language complexity measurement tasks, through the computation of the LIX readability metric and Average Dependency Distance (ADD). Using Swedish high school and university-level essays, we evaluate the models' abilities to compute LIX scores and perform dependency parsing, comparing their results to established ground truths. Our findings reveal that while all models demonstrate some capacity for these tasks, ChatGPT-o1-mini performs most consistently, achieving the highest accuracy in both LIX computation and dependency parsing. Additionally, we observe a strong significant correlation -0.875 p 0.026 (N=6) between the models' accuracy in computing LIX and their overall performance on the Massive Multitask Language Understanding (MMLU) benchmark. These results suggest that language complexity measurement abilities can serve as a noisy zero-shot proxies for assessing the general capabilities of LLMs, providing a practical method for model evaluation without the need for extensive benchmarking datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16205v4">LLMs can be Dangerous Reasoners: Analyzing-based Jailbreak Attack on Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      The rapid development of Large Language Models (LLMs) has brought significant advancements across various tasks. However, despite these achievements, LLMs still exhibit inherent safety vulnerabilities, especially when confronted with jailbreak attacks. Existing jailbreak methods suffer from two main limitations: reliance on complicated prompt engineering and iterative optimization, which lead to low attack success rate (ASR) and attack efficiency (AE). In this work, we propose an efficient jailbreak attack method, Analyzing-based Jailbreak (ABJ), which leverages the advanced reasoning capability of LLMs to autonomously generate harmful content, revealing their underlying safety vulnerabilities during complex reasoning process. We conduct comprehensive experiments on ABJ across various open-source and closed-source LLMs. In particular, ABJ achieves high ASR (82.1% on GPT-4o-2024-11-20) with exceptional AE among all target LLMs, showcasing its remarkable attack effectiveness, transferability, and efficiency. Our findings underscore the urgent need to prioritize and improve the safety of LLMs to mitigate the risks of misuse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23166v2">SciPIP: An LLM-based Scientific Paper Idea Proposer</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 20 pages, 5 figures, 12 tables. The code has been availabel: https://github.com/cheerss/SciPIP
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for automating the proposal of innovative scientific ideas. This process involves two key phases: literature retrieval and idea generation. However, existing approaches often fall short due to their reliance on keyword-based search tools during the retrieval phase, which neglects crucial semantic information and frequently results in incomplete retrieval outcomes. Similarly, in the idea generation phase, current methodologies tend to depend solely on the internal knowledge of LLMs or metadata from retrieved papers, thereby overlooking significant valuable insights contained within the full texts. To address these limitations, we introduce SciPIP, an innovative framework designed to enhance the LLM-based proposal of scientific ideas through improvements in both literature retrieval and idea generation. Our approach begins with the construction of a comprehensive literature database that supports advanced retrieval based not only on keywords but also on semantics and citation relationships. This is complemented by the introduction of a multi-granularity retrieval algorithm aimed at ensuring more thorough and exhaustive retrieval results. For the idea generation phase, we propose a dual-path framework that effectively integrates both the content of retrieved papers and the extensive internal knowledge of LLMs. This integration significantly boosts the novelty, feasibility, and practical value of proposed ideas. Our experiments, conducted across various domains such as natural language processing and computer vision, demonstrate SciPIP's capability to generate a multitude of innovative and useful ideas. These findings underscore SciPIP's potential as a valuable tool for researchers seeking to advance their fields with groundbreaking concepts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22662v2">EMOS: Embodiment-aware Heterogeneous Multi-robot Operating System with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 10 pages of main content, 3 pages of references, 5 pages of appendix, 7 figures in total
    </div>
    <details class="paper-abstract">
      Heterogeneous multi-robot systems (HMRS) have emerged as a powerful approach for tackling complex tasks that single robots cannot manage alone. Current large-language-model-based multi-agent systems (LLM-based MAS) have shown success in areas like software development and operating systems, but applying these systems to robot control presents unique challenges. In particular, the capabilities of each agent in a multi-robot system are inherently tied to the physical composition of the robots, rather than predefined roles. To address this issue, we introduce a novel multi-agent framework designed to enable effective collaboration among heterogeneous robots with varying embodiments and capabilities, along with a new benchmark named Habitat-MAS. One of our key designs is $\textit{Robot Resume}$: Instead of adopting human-designed role play, we propose a self-prompted approach, where agents comprehend robot URDF files and call robot kinematics tools to generate descriptions of their physics capabilities to guide their behavior in task planning and action execution. The Habitat-MAS benchmark is designed to assess how a multi-agent framework handles tasks that require embodiment-aware reasoning, which includes 1) manipulation, 2) perception, 3) navigation, and 4) comprehensive multi-floor object rearrangement. The experimental results indicate that the robot's resume and the hierarchical design of our multi-agent system are essential for the effective operation of the heterogeneous multi-robot system within this intricate problem context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11544v1">Evaluating o1-Like LLMs: Unlocking Reasoning for Translation through Comprehensive Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      The o1-Like LLMs are transforming AI by simulating human cognitive processes, but their performance in multilingual machine translation (MMT) remains underexplored. This study examines: (1) how o1-Like LLMs perform in MMT tasks and (2) what factors influence their translation quality. We evaluate multiple o1-Like LLMs and compare them with traditional models like ChatGPT and GPT-4o. Results show that o1-Like LLMs establish new multilingual translation benchmarks, with DeepSeek-R1 surpassing GPT-4o in contextless tasks. They demonstrate strengths in historical and cultural translation but exhibit a tendency for rambling issues in Chinese-centric outputs. Further analysis reveals three key insights: (1) High inference costs and slower processing speeds make complex translation tasks more resource-intensive. (2) Translation quality improves with model size, enhancing commonsense reasoning and cultural translation. (3) The temperature parameter significantly impacts output quality-lower temperatures yield more stable and accurate translations, while higher temperatures reduce coherence and precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03138v2">Can LLMs Generate Diverse Molecules? Towards Alignment with Structural Diversity</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated impressive performance in molecular generation, which offers potential to accelerate drug discovery. However, the current LLMs overlook a critical requirement for drug discovery: proposing a diverse set of molecules. This diversity is essential for improving the chances of finding a viable drug, as it provides alternative molecules that may succeed where others fail in real-world validations. Nevertheless, the LLMs often output structurally similar molecules. While decoding schemes like diverse beam search may enhance textual diversity, this often does not align with molecular structural diversity. In response, we propose a new method for fine-tuning molecular generative LLMs to autoregressively generate a set of structurally diverse molecules, where each molecule is generated by conditioning on the previously generated molecules. Our approach consists of two stages: (1) supervised fine-tuning to adapt LLMs to autoregressively generate molecules in a sequence and (2) reinforcement learning to maximize structural diversity within the generated molecules. Our experiments show that the proposed approach enables LLMs to generate diverse molecules better than existing approaches for diverse sequence generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11686v4">CCoE: A Compact and Efficient LLM Framework with Multi-Expert Collaboration for Resource-Limited Settings</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved exceptional performance across diverse domains through training on massive datasets. However, scaling LLMs to support multiple downstream domain applications remains a significant challenge, especially under resource constraints. Existing approaches often struggle to balance performance across multiple domains with resource efficiency, limiting their broader applicability. To address this, we introduce the CCoE architecture, a modular framework that seamlessly integrates domain-specific experts into a unified LLM. By leveraging independently trained expert subnetworks on a shared backbone partition, CCoE achieves state-of-the-art performance while significantly reducing the resource requirements for multi-expert deployments. Furthermore, rule-based gating and expert planning in CCoE enable flexible task allocation, promoting expert collaboration to handle complex reasoning tasks. CCoE not only reduces inference costs but also provides a flexible and scalable solution for integrating domain expertise across diverse applications. Experiments on five domains demonstrate that CCoE achieves comparable performance to current domain-specific LLMs. Moreover, compared to existing multi-domain model ensemble methods, CCoE reduces memory usage by 61.3%, while improving inference efficiency by 0.76x over parameter-efficient multi-expert integration approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11533v1">Be Cautious When Merging Unfamiliar LLMs: A Phishing Model Capable of Stealing Privacy</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Model merging is a widespread technology in large language models (LLMs) that integrates multiple task-specific LLMs into a unified one, enabling the merged model to inherit the specialized capabilities of these LLMs. Most task-specific LLMs are sourced from open-source communities and have not undergone rigorous auditing, potentially imposing risks in model merging. This paper highlights an overlooked privacy risk: \textit{an unsafe model could compromise the privacy of other LLMs involved in the model merging.} Specifically, we propose PhiMM, a privacy attack approach that trains a phishing model capable of stealing privacy using a crafted privacy phishing instruction dataset. Furthermore, we introduce a novel model cloaking method that mimics a specialized capability to conceal attack intent, luring users into merging the phishing model. Once victims merge the phishing model, the attacker can extract personally identifiable information (PII) or infer membership information (MI) by querying the merged model with the phishing instruction. Experimental results show that merging a phishing model increases the risk of privacy breaches. Compared to the results before merging, PII leakage increased by 3.9\% and MI leakage increased by 17.4\% on average. We release the code of PhiMM through a link.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11521v1">DeFiScope: Detecting Various DeFi Price Manipulations with LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      DeFi (Decentralized Finance) is one of the most important applications of today's cryptocurrencies and smart contracts. It manages hundreds of billions in Total Value Locked (TVL) on-chain, yet it remains susceptible to common DeFi price manipulation attacks. Despite state-of-the-art (SOTA) systems like DeFiRanger and DeFort, we found that they are less effective to non-standard price models in custom DeFi protocols, which account for 44.2% of the 95 DeFi price manipulation attacks reported over the past three years. In this paper, we introduce the first LLM-based approach, DeFiScope, for detecting DeFi price manipulation attacks in both standard and custom price models. Our insight is that large language models (LLMs) have certain intelligence to abstract price calculation from code and infer the trend of token price changes based on the extracted price models. To further strengthen LLMs in this aspect, we leverage Foundry to synthesize on-chain data and use it to fine-tune a DeFi price-specific LLM. Together with the high-level DeFi operations recovered from low-level transaction data, DeFiScope detects various DeFi price manipulations according to systematically mined patterns. Experimental results show that DeFiScope achieves a high precision of 96% and a recall rate of 80%, significantly outperforming SOTA approaches. Moreover, we evaluate DeFiScope's cost-effectiveness and demonstrate its practicality by helping our industry partner confirm 147 real-world price manipulation attacks, including discovering 81 previously unknown historical incidents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12480v2">KcMF: A Knowledge-compliant Framework for Schema and Entity Matching with Fine-tuning-free LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 under reveiw; new results and analysis added, typos corrected
    </div>
    <details class="paper-abstract">
      Schema matching (SM) and entity matching (EM) tasks are crucial for data integration. While large language models (LLMs) have shown promising results in these tasks, they suffer from hallucinations and confusion about task instructions. This study presents the Knowledge-Compliant Matching Framework (KcMF), an LLM-based approach that addresses these issues without the need for domain-specific fine-tuning. KcMF employs a once-and-for-all pseudo-code-based task decomposition strategy to adopt natural language statements that guide LLM reasoning and reduce confusion across various task types. We also propose two mechanisms, Dataset as Knowledge (DaK) and Example as Knowledge (EaK), to build domain knowledge sets when unstructured domain knowledge is lacking. Moreover, we introduce a result-ensemble strategy to leverage multiple knowledge sources and suppress badly formatted outputs. Extensive evaluations confirm that KcMF clearly enhances five LLM backbones in both SM and EM tasks while outperforming the non-LLM competitors by an average F1-score of 17.93%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14744v4">Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to face hallucination issues due to the data they trained on often containing human bias; whether this is reflected in the decision-making process of LLM Agents remains under-explored. As LLM Agents are increasingly employed in intricate social environments, a pressing and natural question emerges: Can we utilize LLM Agents' systematic hallucinations to mirror human cognitive biases, thus exhibiting irrational social intelligence? In this paper, we probe the irrational behavior among contemporary LLM Agents by melding practical social science experiments with theoretical insights. Specifically, We propose CogMir, an open-ended Multi-LLM Agents framework that utilizes hallucination properties to assess and enhance LLM Agents' social intelligence through cognitive biases. Experimental results on CogMir subsets show that LLM Agents and humans exhibit high consistency in irrational and prosocial decision-making under uncertain conditions, underscoring the prosociality of LLM Agents as social entities and highlighting the significance of hallucination properties. Additionally, the CogMir framework demonstrates its potential as a valuable platform for encouraging more research into the social intelligence of LLM Agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16756v3">How Well Do LLMs Handle Cantonese? Benchmarking Cantonese Capabilities of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted by NAACL 2025
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has transformed the competitive landscape in natural language processing (NLP), particularly for English and other data-rich languages. However, underrepresented languages like Cantonese, spoken by over 85 million people, face significant development gaps, which is particularly concerning given the economic significance of the Guangdong-Hong Kong-Macau Greater Bay Area, and in substantial Cantonese-speaking populations in places like Singapore and North America. Despite its wide use, Cantonese has scant representation in NLP research, especially compared to other languages from similarly developed regions. To bridge these gaps, we outline current Cantonese NLP methods and introduce new benchmarks designed to evaluate LLM performance in factual generation, mathematical logic, complex reasoning, and general knowledge in Cantonese, which aim to advance open-source Cantonese LLM technology. We also propose future research directions and recommended models to enhance Cantonese LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11493v1">DAST: Context-Aware Compression in LLMs via Dynamic Allocation of Soft Tokens</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) face computational inefficiencies and redundant processing when handling long context inputs, prompting a focus on compression techniques. While existing semantic vector-based compression methods achieve promising performance, these methods fail to account for the intrinsic information density variations between context chunks, instead allocating soft tokens uniformly across context chunks. This uniform distribution inevitably diminishes allocation to information-critical regions. To address this, we propose Dynamic Allocation of Soft Tokens (DAST), a simple yet effective method that leverages the LLM's intrinsic understanding of contextual relevance to guide compression. DAST combines perplexity-based local information with attention-driven global information to dynamically allocate soft tokens to the informative-rich chunks, enabling effective, context-aware compression. Experimental results across multiple benchmarks demonstrate that DAST surpasses state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15549v2">WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance, aligning these models with human preferences has emerged as a critical challenge. Traditional alignment methods, relying on human or LLM annotated datasets, are limited by their resource-intensive nature, inherent subjectivity, misalignment with real-world user preferences, and the risk of feedback loops that amplify model biases. To overcome these limitations, we introduce WildFeedback, a novel framework that leverages in-situ user feedback during conversations with LLMs to create preference datasets automatically. Given a corpus of multi-turn user-LLM conversation, WildFeedback identifies and classifies user feedback to LLM responses between conversation turns. The user feedback is then used to create examples of preferred and dispreferred responses according to users' preference. Our experiments demonstrate that LLMs fine-tuned on WildFeedback dataset exhibit significantly improved alignment with user preferences, as evidenced by both traditional benchmarks and our proposed checklist-guided evaluation. By incorporating in-situ feedback from actual users, WildFeedback addresses the scalability, subjectivity, and bias challenges that plague existing approaches, marking a significant step toward developing LLMs that are more responsive to the diverse and evolving needs of their users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11471v1">GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Knowledge Graph Completion (KGC), which aims to infer missing or incomplete facts, is a crucial task for KGs. However, integrating the vital structural information of KGs into Large Language Models (LLMs) and outputting predictions deterministically remains challenging. To address this, we propose a new method called GLTW, which encodes the structural information of KGs and merges it with LLMs to enhance KGC performance. Specifically, we introduce an improved Graph Transformer (iGT) that effectively encodes subgraphs with both local and global structural information and inherits the characteristics of language model, bypassing training from scratch. Also, we develop a subgraph-based multi-classification training objective, using all entities within KG as classification objects, to boost learning efficiency.Importantly, we combine iGT with an LLM that takes KG language prompts as input.Our extensive experiments on various KG datasets show that GLTW achieves significant performance gains compared to SOTA baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09077v2">CuriousLLM: Elevating Multi-Document Question Answering with LLM-Enhanced Knowledge Graph Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant success in open-domain question answering. However, they continue to face challenges such as hallucinations and knowledge cutoffs. These issues can be mitigated through in-context learning by providing LLMs with relevant context before generating answers. Recent literature proposes Knowledge Graph Prompting (KGP) which integrates knowledge graphs with an LLM-based traversal agent to substantially enhance document retrieval quality. However, KGP requires costly fine-tuning with large datasets and remains prone to hallucination. In this paper, we propose CuriousLLM, an enhancement that integrates a curiosity-driven reasoning mechanism into an LLM agent. This mechanism enables the agent to generate relevant follow-up questions, thereby guiding the information retrieval process more efficiently. Central to our approach is the development of the new Follow-upQA dataset, which includes questions and supporting evidence as input, with follow-up questions serving as ground truths. These follow-up questions either inquire about what is still missing to fully answer the user's query or use special tokens to signify that the retrieved evidence is sufficient. Our experiments show that CuriousLLM significantly boosts LLM performance in multi-document question answering (MD-QA), circumventing the substantial computational costs and latency from the original KGP framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11451v1">From Personas to Talks: Revisiting the Impact of Personas on LLM-Synthesized Emotional Support Conversations</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has revolutionized the generation of emotional support conversations (ESC), offering scalable solutions with reduced costs and enhanced data privacy. This paper explores the role of personas in the creation of ESC by LLMs. Our research utilizes established psychological frameworks to measure and infuse persona traits into LLMs, which then generate dialogues in the emotional support scenario. We conduct extensive evaluations to understand the stability of persona traits in dialogues, examining shifts in traits post-generation and their impact on dialogue quality and strategy distribution. Experimental results reveal several notable findings: 1) LLMs can infer core persona traits, 2) subtle shifts in emotionality and extraversion occur, influencing the dialogue dynamics, and 3) the application of persona traits modifies the distribution of emotional support strategies, enhancing the relevance and empathetic quality of the responses. These findings highlight the potential of persona-driven LLMs in crafting more personalized, empathetic, and effective emotional support dialogues, which has significant implications for the future design of AI-driven emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11441v1">Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) risk retaining unauthorized or sensitive information from their training data, which raises privacy concerns. LLM unlearning seeks to mitigate these risks by selectively removing specified data while maintaining overall model performance. However, most existing work focus on methods to achieve effective forgetting and does not provide a detailed analysis of the retain set, the portion of training data that is not targeted for removal. In this paper, we investigate the effects of unlearning on various subsets of the retain set through a case study on entity unlearning. We introduce the Syntactically Similar Neighbor Set, a group of queries that share similar syntactic structures with the data targeted for removal, and show that this subset suffers the greatest performance drop during unlearning. Moreover, when used for regularization, this set not only preserves performance on syntactically similar queries but also delivers comparable or improved results across other data subsets. Our results highlight that syntactic similarity is a critical factor, potentially more so than domain or entity relationships, in achieving effective and practical LLM unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11436v1">ADO: Automatic Data Optimization for Inputs in LLM Prompts</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      This study explores a novel approach to enhance the performance of Large Language Models (LLMs) through the optimization of input data within prompts. While previous research has primarily focused on refining instruction components and augmenting input data with in-context examples, our work investigates the potential benefits of optimizing the input data itself. We introduce a two-pronged strategy for input data optimization: content engineering and structural reformulation. Content engineering involves imputing missing values, removing irrelevant attributes, and enriching profiles by generating additional information inferred from existing attributes. Subsequent to content engineering, structural reformulation is applied to optimize the presentation of the modified content to LLMs, given their sensitivity to input format. Our findings suggest that these optimizations can significantly improve the performance of LLMs in various tasks, offering a promising avenue for future research in prompt engineering. The source code is available at https://anonymous.4open.science/r/ADO-6BC5/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11433v1">\textsc{FLAG-Trader}: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) fine-tuned on multimodal financial data have demonstrated impressive reasoning capabilities in various financial tasks. However, they often struggle with multi-step, goal-oriented scenarios in interactive financial markets, such as trading, where complex agentic approaches are required to improve decision-making. To address this, we propose \textsc{FLAG-Trader}, a unified architecture integrating linguistic processing (via LLMs) with gradient-driven reinforcement learning (RL) policy optimization, in which a partially fine-tuned LLM acts as the policy network, leveraging pre-trained knowledge while adapting to the financial domain through parameter-efficient fine-tuning. Through policy gradient optimization driven by trading rewards, our framework not only enhances LLM performance in trading but also improves results on other financial-domain tasks. We present extensive empirical evidence to validate these enhancements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05551v2">FRAMES: Boosting LLMs with A Four-Quadrant Multi-Stage Pretraining Strategy</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced human language understanding and generation, with pretraining data quality and organization being crucial to their performance. Multi-stage pretraining is a promising approach, but existing methods often lack quantitative criteria for data partitioning and instead rely on intuitive heuristics. In this paper, we propose the novel Four-quadRAnt Multi-stage prEtraining strategy (FRAME), guided by the established principle of organizing the pretraining process into four stages to achieve significant loss reductions four times. This principle is grounded in two key findings: first, training on high Perplexity (PPL) data followed by low PPL data, and second, training on low PPL difference (PD) data followed by high PD data, both causing the loss to drop significantly twice and performance enhancements. By partitioning data into four quadrants and strategically organizing them, FRAME achieves a remarkable 16.8% average improvement over random across MMLU and CMMLU for the 3B model, effectively boosting LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14141v2">LLMs can Realize Combinatorial Creativity: Generating Creative Ideas via LLMs for Scientific Research</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Scientific idea generation has been extensively studied in creativity theory and computational creativity research, providing valuable frameworks for understanding and implementing creative processes. However, recent work using Large Language Models (LLMs) for research idea generation often overlooks these theoretical foundations. We present a framework that explicitly implements combinatorial creativity theory using LLMs, featuring a generalization-level retrieval system for cross-domain knowledge discovery and a structured combinatorial process for idea generation. The retrieval system maps concepts across different abstraction levels to enable meaningful connections between disparate domains, while the combinatorial process systematically analyzes and recombines components to generate novel solutions. Experiments on the OAG-Bench dataset demonstrate our framework's effectiveness, consistently outperforming baseline approaches in generating ideas that align with real research developments (improving similarity scores by 7\%-10\% across multiple metrics). Our results provide strong evidence that LLMs can effectively realize combinatorial creativity when guided by appropriate theoretical frameworks, contributing both to practical advancement of AI-assisted research and theoretical understanding of machine creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11417v1">DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 17 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The rapid rise of large language models (LLMs) in text streaming services has introduced significant cost and Quality of Experience (QoE) challenges in serving millions of daily requests, especially in meeting Time-To-First-Token (TTFT) and Time-Between-Token (TBT) requirements for real-time interactions. Our real-world measurements show that both server-based and on-device deployments struggle to meet diverse QoE demands: server deployments face high costs and last-hop issues (e.g., Internet latency and dynamics), while on-device LLM inference is constrained by resources. We introduce DiSCo, a device-server cooperative scheduler designed to optimize users' QoE by adaptively routing requests and migrating response generation between endpoints while maintaining cost constraints. DiSCo employs cost-aware scheduling, leveraging the predictable speed of on-device LLM inference with the flexible capacity of server-based inference to dispatch requests on the fly, while introducing a token-level migration mechanism to ensure consistent token delivery during migration. Evaluations on real-world workloads -- including commercial services like OpenAI GPT and DeepSeek, and open-source deployments such as LLaMA3 -- show that DiSCo can improve users' QoE by reducing tail TTFT (11-52\%) and mean TTFT (6-78\%) across different model-device configurations, while dramatically reducing serving costs by up to 84\% through its migration mechanism while maintaining comparable QoE levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13949v2">Mufu: Multilingual Fused Learning for Low-Resource Translation with LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) are great translators, but this is largely limited to high-resource languages. For many LLMs, translating in and out of low-resource languages remains a challenging task. To maximize data efficiency in this low-resource setting, we introduce Mufu, which includes a selection of automatically generated multilingual candidates and an instruction to correct inaccurate translations in the prompt. Mufu prompts turn a translation task into a postediting one, and seek to harness the LLM's reasoning capability with auxiliary translation candidates, from which the model is required to assess the input quality, align the semantics cross-lingually, copy from relevant inputs and override instances that are incorrect. Our experiments on En-XX translations over the Flores-200 dataset show LLMs finetuned against Mufu-style prompts are robust to poor quality auxiliary translation candidates, achieving performance superior to NLLB 1.3B distilled model in 64% of low- and very-low-resource language pairs. We then distill these models to reduce inference cost, while maintaining on average 3.1 chrF improvement over finetune-only baseline in low-resource translations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07191v4">Bag of Tricks for Inference-time Computation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      With the advancement of large language models (LLMs), solving complex reasoning tasks has gained increasing attention. Inference-time computation methods (e.g., Best-of-N, beam search, et al.) are particularly valuable as they can enhance reasoning performance without modifying model parameters or requiring additional training. However, these techniques come with implementation challenges, and most existing methods remain at the proof-of-concept stage with limited practical adoption due to their computational complexity and varying effectiveness across different tasks. In this paper, we investigate and benchmark diverse inference-time computation strategies across reasoning tasks of varying complexity. Since most current methods rely on a proposer-verifier pipeline that first generates candidate solutions (e.g., reasoning solutions) and then selects the best one based on reward signals (e.g., RLHF rewards, process rewards), our research focuses on optimizing both candidate solution generation (e.g., instructing prompts, hyperparameters such as temperature and top-p) and reward mechanisms (e.g., self-evaluation, reward types). Through extensive experiments (more than 20,000 A100-80G GPU hours with over 1,000 experiments) across a variety of models (e.g., Llama, Qwen, and Mistral families) of various sizes, our ablation studies reveal that previously overlooked strategies can significantly enhance performance (e.g., tuning temperature can improve reasoning task performance by up to 5%). Furthermore, we establish a standardized benchmark for inference-time computation by systematically evaluating six representative methods across eight reasoning tasks. These findings provide a stronger foundation for future research. The code is available at https://github.com/usail-hkust/benchmark_inference_time_computation_LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02795v3">InfiFusion: A Unified Framework for Enhanced Cross-Model Reasoning via LLM Fusion</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Significant performance improvements over the previous version; under review;
    </div>
    <details class="paper-abstract">
      We introduce InfiFusion, an efficient training pipeline designed to integrate multiple domain-specialized Large Language Models (LLMs) into a single pivot model, effectively harnessing the strengths of each source model. Traditional fusion methods either merge model parameters directly or rely on knowledge distillation with rigid assumptions, limiting their flexibility and efficiency. InfiFusion overcomes these limitations by enhancing Universal Logit Distillation (ULD) with Top-K selection and Logits Standardization. We propose two fusion strategies: Pairwise Fusion (InfiFusion$_p$), where each source model knowledge is distilled individually into the pivot model followed by merging and Unified Fusion (InfiFusion$_u$), where knowledge from all source models is distilled simultaneously into the pivot model. InfiFusion outperforms the state-of-the-art models, such as Qwen-2.5-14B-Instruct and Phi-4, across 11 widely applied benchmarks covering reasoning, coding, mathematics, and instruction-following tasks. Notably, InfiFusion achieves this superior performance while significantly reduces computational costs, completing full training with only 160 H800 GPU hours compared to the millions typically required for traditional LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09412v2">FB-Bench: A Fine-Grained Multi-Task Benchmark for Evaluating LLMs' Responsiveness to Human Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Human feedback is crucial in the interactions between humans and Large Language Models (LLMs). However, existing research primarily focuses on benchmarking LLMs in single-turn dialogues. Even in benchmarks designed for multi-turn dialogues, the user inputs are often independent, neglecting the nuanced and complex nature of human feedback within real-world usage scenarios. To fill this research gap, we introduce FB-Bench, a fine-grained, multi-task benchmark designed to evaluate LLMs' responsiveness to human feedback under real-world usage scenarios in Chinese. Drawing from the two main interaction scenarios, FB-Bench comprises 591 meticulously curated samples, encompassing eight task types, five deficiency types of response, and nine feedback types. We extensively evaluate a broad array of popular LLMs, revealing significant variations in their performance across different interaction scenarios. Further analysis indicates that task, human feedback, and deficiencies of previous responses can also significantly impact LLMs' responsiveness. Our findings underscore both the strengths and limitations of current models, providing valuable insights and directions for future research. Code and datasets are available at https://github.com/PKU-Baichuan-MLSystemLab/FB-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11401v1">Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      A new trend uses LLMs as dense text encoders via contrastive learning. However, since LLM embeddings predict the probability distribution of the next token, they are inherently generative and distributive, conflicting with contrastive learning, which requires embeddings to capture full-text semantics and align via cosine similarity. This discrepancy hinders the full utilization of LLMs' pre-training capabilities, resulting in inefficient learning. In response to this issue, we propose AutoRegEmbed, a new contrastive learning method built on embedding conditional probability distributions, which integrates two core tasks: information compression and conditional distribution alignment. The information compression task encodes text into the embedding space, ensuring that the embedding vectors capture global semantics. The conditional distribution alignment task focuses on aligning text embeddings with positive samples embeddings by leveraging the conditional distribution of embeddings while simultaneously reducing the likelihood of generating negative samples from text embeddings, thereby achieving embedding alignment and uniformity. Experimental results demonstrate that our method significantly outperforms traditional contrastive learning approaches and achieves performance comparable to state-of-the-art models when using the same amount of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11400v1">Revisiting Robust RAG: Do We Still Need Complex Robust Training in the Era of Powerful LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) systems often suffer from performance degradation when encountering noisy or irrelevant documents, driving researchers to develop sophisticated training strategies to enhance their robustness against such retrieval noise. However, as large language models (LLMs) continue to advance, the necessity of these complex training methods is increasingly questioned. In this paper, we systematically investigate whether complex robust training strategies remain necessary as model capacity grows. Through comprehensive experiments spanning multiple model architectures and parameter scales, we evaluate various document selection methods and adversarial training techniques across diverse datasets. Our extensive experiments consistently demonstrate that as models become more powerful, the performance gains brought by complex robust training methods drop off dramatically. We delve into the rationale and find that more powerful models inherently exhibit superior confidence calibration, better generalization across datasets (even when trained with randomly selected documents), and optimal attention mechanisms learned with simpler strategies. Our findings suggest that RAG systems can benefit from simpler architectures and training strategies as models become more powerful, enabling more scalable applications with minimal complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12445v2">Open Ko-LLM Leaderboard2: Bridging Foundational and Practical Evaluation for Korean LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted to NAACL 2025 Industry
    </div>
    <details class="paper-abstract">
      The Open Ko-LLM Leaderboard has been instrumental in benchmarking Korean Large Language Models (LLMs), yet it has certain limitations. Notably, the disconnect between quantitative improvements on the overly academic leaderboard benchmarks and the qualitative impact of the models should be addressed. Furthermore, the benchmark suite is largely composed of translated versions of their English counterparts, which may not fully capture the intricacies of the Korean language. To address these issues, we propose Open Ko-LLM Leaderboard2, an improved version of the earlier Open Ko-LLM Leaderboard. The original benchmarks are entirely replaced with new tasks that are more closely aligned with real-world capabilities. Additionally, four new native Korean benchmarks are introduced to better reflect the distinct characteristics of the Korean language. Through these refinements, Open Ko-LLM Leaderboard2 seeks to provide a more meaningful evaluation for advancing Korean LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11393v1">HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in commonsense reasoning; however, some variations in questions can trigger incorrect responses. Do these models truly understand commonsense knowledge, or just memorize expression patterns? To investigate this question, we present the first extensive robustness evaluation of LLMs in commonsense reasoning. We introduce HellaSwag-Pro, a large-scale bilingual benchmark consisting of 11,200 cases, by designing and compiling seven types of question variants. To construct this benchmark, we propose a two-stage method to develop Chinese HellaSwag, a finely annotated dataset comprising 12,000 instances across 56 categories. We conduct extensive experiments on 41 representative LLMs, revealing that these LLMs are far from robust in commonsense reasoning. Furthermore, this robustness varies depending on the language in which the LLM is tested. This work establishes a high-quality evaluation benchmark, with extensive experiments offering valuable insights to the community in commonsense reasoning for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00804v2">Examining Identity Drift in Conversations of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show impressive conversational abilities but sometimes show identity drift problems, where their interaction patterns or styles change over time. As the problem has not been thoroughly examined yet, this study examines identity consistency across nine LLMs. Specifically, we (1) investigate whether LLMs could maintain consistent patterns (or identity) and (2) analyze the effect of the model family, parameter sizes, and provided persona types. Our experiments involve multi-turn conversations on personal themes, analyzed in qualitative and quantitative ways. Experimental results indicate three findings. (1) Larger models experience greater identity drift. (2) Model differences exist, but their effect is not stronger than parameter sizes. (3) Assigning a persona may not help to maintain identity. We hope these three findings can help to improve persona stability in AI-driven dialogue systems, particularly in long-term conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11380v1">Exploring the Small World of Word Embeddings: A Comparative Study on Conceptual Spaces from LLMs of Different Scales</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Paper under review
    </div>
    <details class="paper-abstract">
      A conceptual space represents concepts as nodes and semantic relatedness as edges. Word embeddings, combined with a similarity metric, provide an effective approach to constructing such a space. Typically, embeddings are derived from traditional distributed models or encoder-only pretrained models, whose objectives directly capture the meaning of the current token. In contrast, decoder-only models, including large language models (LLMs), predict the next token, making their embeddings less directly tied to the current token's semantics. Moreover, comparative studies on LLMs of different scales remain underexplored. In this paper, we construct a conceptual space using word embeddings from LLMs of varying scales and comparatively analyze their properties. We establish a network based on a linguistic typology-inspired connectivity hypothesis, examine global statistical properties, and compare LLMs of varying scales. Locally, we analyze conceptual pairs, WordNet relations, and a cross-lingual semantic network for qualitative words. Our results indicate that the constructed space exhibits small-world properties, characterized by a high clustering coefficient and short path lengths. Larger LLMs generate more intricate spaces, with longer paths reflecting richer relational structures and connections. Furthermore, the network serves as an efficient bridge for cross-lingual semantic mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01558v2">Predicting the Performance of Black-box LLMs through Self-Queries</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly relied on in AI systems, predicting when they make mistakes is crucial. While a great deal of work in the field uses internal representations to interpret model behavior, these representations are inaccessible when given solely black-box access through an API. In this paper, we extract features of LLMs in a black-box manner by using follow-up prompts and taking the probabilities of different responses as representations to train reliable predictors of model behavior. We demonstrate that training a linear model on these low-dimensional representations produces reliable and generalizable predictors of model performance at the instance level (e.g., if a particular generation correctly answers a question). Remarkably, these can often outperform white-box linear predictors that operate over a model's hidden state or the full distribution over its vocabulary. In addition, we demonstrate that these extracted features can be used to evaluate more nuanced aspects of a language model's state. For instance, they can be used to distinguish between a clean version of GPT-4o-mini and a version that has been influenced via an adversarial system prompt that answers question-answering tasks incorrectly or introduces bugs into generated code. Furthermore, they can reliably distinguish between different model architectures and sizes, enabling the detection of misrepresented models provided through an API (e.g., identifying if GPT-3.5 is supplied instead of GPT-4o-mini).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11368v1">LLMs can Perform Multi-Dimensional Analytic Writing Assessments: A Case Study of L2 Graduate-Level Academic English Writing</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 26 pages, 6 figures, 15 tables
    </div>
    <details class="paper-abstract">
      The paper explores the performance of LLMs in the context of multi-dimensional analytic writing assessments, i.e. their ability to provide both scores and comments based on multiple assessment criteria. Using a corpus of literature reviews written by L2 graduate students and assessed by human experts against 9 analytic criteria, we prompt several popular LLMs to perform the same task under various conditions. To evaluate the quality of feedback comments, we apply a novel feedback comment quality evaluation framework. This framework is interpretable, cost-efficient, scalable, and reproducible, compared to existing methods that rely on manual judgments. We find that LLMs can generate reasonably good and generally reliable multi-dimensional analytic assessments. We release our corpus for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13276v4">SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Attention is the cornerstone of modern Large Language Models (LLMs). Yet its quadratic complexity hinders efficiency and scalability, especially for long-context processing. A promising approach is to leverage sparsity in attention. However, existing sparsity-based solutions predominantly rely on predefined patterns or heuristics at the attention head level, struggling to adapt dynamically to different contexts efficiently. We propose SeerAttention, a simple yet effective attention mechanism that directly learns the block-level attention sparsity from the LLM itself. Inspired by the gating mechanism in Mixture of Experts (MoE), SeerAttention augments the conventional attention with a learnable gate that selectively activates important blocks within the attention map. Specifically, the gate first pools the query (Q) and key (K) tensors along the sequence dimension and processes them through learnable linear layers. The resulting matrices are then multiplied together to produce the gating scores, which are used to predict block-level attention sparsity. Combined with our block-sparse FlashAttention kernel, SeerAttention can achieve significant speedup on GPUs. When applied to pre-trained LLMs, SeerAttention only requires training the gate parameters in a lightweight self-distillation manner, allowing rapid convergence. Our evaluation results demonstrate that SeerAttention achieves better model accuracy and lower latency for long-context pre-filling compared to prior methods. Code is available at: https://github.com/microsoft/SeerAttention
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13115v2">Dagger Behind Smile: Fool LLMs with a Happy Ending Story</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      The wide adoption of Large Language Models (LLMs) has attracted significant attention from $\textit{jailbreak}$ attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious contents. However, optimization-based attacks have limited efficiency and transferability, while existing manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to $\textit{positive}$ prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a $\textit{happy ending}$, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request.This has made HEA both efficient and effective, as it requires only up to two turns to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% attack success rate on average. We also provide quantitative explanations for the success of HEA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11358v1">Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 15 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11355v1">"Nuclear Deployed!": Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Our code will be available at https://github.com/pillowsofwind/LLM-CBRN-Risks
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are evolving into autonomous decision-makers, raising concerns about catastrophic risks in high-stakes scenarios, particularly in Chemical, Biological, Radiological and Nuclear (CBRN) domains. Based on the insight that such risks can originate from trade-offs between the agent's Helpful, Harmlessness and Honest (HHH) goals, we build a novel three-stage evaluation framework, which is carefully constructed to effectively and naturally expose such risks. We conduct 14,400 agentic simulations across 12 advanced LLMs, with extensive experiments and analysis. Results reveal that LLM agents can autonomously engage in catastrophic behaviors and deception, without being deliberately induced. Furthermore, stronger reasoning abilities often increase, rather than mitigate, these risks. We also show that these agents can violate instructions and superior commands. On the whole, we empirically prove the existence of catastrophic risks in autonomous LLM agents. We will release our code upon request.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03257v2">Understanding LLM Development Through Longitudinal Study: Insights from the Open Ko-LLM Leaderboard</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted to NAACL 2025 Industry
    </div>
    <details class="paper-abstract">
      This paper conducts a longitudinal study over eleven months to address the limitations of prior research on the Open Ko-LLM Leaderboard, which have relied on empirical studies with restricted observation periods of only five months. By extending the analysis duration, we aim to provide a more comprehensive understanding of the progression in developing Korean large language models (LLMs). Our study is guided by three primary research questions: (1) What are the specific challenges in improving LLM performance across diverse tasks on the Open Ko-LLM Leaderboard over time? (2) How does model size impact task performance correlations across various benchmarks? (3) How have the patterns in leaderboard rankings shifted over time on the Open Ko-LLM Leaderboard?. By analyzing 1,769 models over this period, our research offers a comprehensive examination of the ongoing advancements in LLMs and the evolving nature of evaluation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12372v1">Factual Inconsistency in Data-to-Text Generation Scales Exponentially with LLM Size: A Statistical Validation</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Monitoring factual inconsistency is essential for ensuring trustworthiness in data-to-text generation (D2T). While large language models (LLMs) have demonstrated exceptional performance across various D2T tasks, previous studies on scaling laws have primarily focused on generalization error through power law scaling to LLM size (i.e., the number of model parameters). However, no research has examined the impact of LLM size on factual inconsistency in D2T. In this paper, we investigate how factual inconsistency in D2T scales with LLM size by exploring two scaling laws: power law and exponential scaling. To rigorously evaluate and compare these scaling laws, we employ a statistical validation framework consisting of three key stages: predictive performance estimation, goodness-of-fit assessment, and comparative analysis. For a comprehensive empirical study, we analyze three popular LLM families across five D2T datasets, measuring factual inconsistency inversely using four state-of-the-art consistency metrics. Our findings, based on exhaustive empirical results and validated through our framework, reveal that, contrary to the widely assumed power law scaling, factual inconsistency in D2T follows an exponential scaling with LLM size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12340v1">Understanding Silent Data Corruption in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      As the scale of training large language models (LLMs) increases, one emergent failure is silent data corruption (SDC), where hardware produces incorrect computations without explicit failure signals. In this work, we are the first to investigate the impact of real-world SDCs on LLM training by comparing model training between healthy production nodes and unhealthy nodes exhibiting SDCs. With the help from a cloud computing platform, we access the unhealthy nodes that were swept out from production by automated fleet management. Using deterministic execution via XLA compiler and our proposed synchronization mechanisms, we isolate and analyze the impact of SDC errors on these nodes at three levels: at each submodule computation, at a single optimizer step, and at a training period. Our results reveal that the impact of SDCs on computation varies on different unhealthy nodes. Although in most cases the perturbations from SDCs on submodule computation and gradients are relatively small, SDCs can lead models to converge to different optima with different weights and even cause spikes in the training loss. Our analysis sheds light on further understanding and mitigating the impact of SDCs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12325v1">From Dense to Dynamic: Token-Difficulty Driven MoEfication of Pre-Trained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) for different inference constraints is computationally expensive, limiting control over efficiency-accuracy trade-offs. Moreover, once trained, these models typically process tokens uniformly, regardless of their complexity, leading to static and inflexible behavior. In this paper, we introduce a post-training optimization framework, DynaMoE, that adapts a pre-trained dense LLM to a token-difficulty-driven Mixture-of-Experts model with minimal fine-tuning cost. This adaptation makes the model dynamic, with sensitivity control to customize the balance between efficiency and accuracy. DynaMoE features a token-difficulty-aware router that predicts the difficulty of tokens and directs them to the appropriate sub-networks or experts, enabling larger experts to handle more complex tokens and smaller experts to process simpler ones. Our experiments demonstrate that DynaMoE can generate a range of adaptive model variants of the existing trained LLM with a single fine-tuning step, utilizing only $10B$ tokens, a minimal cost compared to the base model's training. Each variant offers distinct trade-offs between accuracy and performance. Compared to the baseline post-training optimization framework, Flextron, our method achieves similar aggregated accuracy across downstream tasks, despite using only $\frac{1}{9}\text{th}$ of their fine-tuning cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12275v1">Integrating Expert Knowledge into Logical Programs via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      This paper introduces ExKLoP, a novel framework designed to evaluate how effectively Large Language Models (LLMs) integrate expert knowledge into logical reasoning systems. This capability is especially valuable in engineering, where expert knowledge-such as manufacturer-recommended operational ranges-can be directly embedded into automated monitoring systems. By mirroring expert verification steps, tasks like range checking and constraint validation help ensure system safety and reliability. Our approach systematically evaluates LLM-generated logical rules, assessing both syntactic fluency and logical correctness in these critical validation tasks. We also explore the models capacity for self-correction via an iterative feedback loop based on code execution outcomes. ExKLoP presents an extensible dataset comprising 130 engineering premises, 950 prompts, and corresponding validation points. It enables comprehensive benchmarking while allowing control over task complexity and scalability of experiments. We leverage the synthetic data creation methodology to conduct extensive empirical evaluation on a diverse set of LLMs including Llama3, Gemma, Mixtral, Mistral, and Qwen. Results reveal that while models generate nearly perfect syntactically correct code, they frequently exhibit logical errors in translating expert knowledge. Furthermore, iterative self-correction yields only marginal improvements (up to 3%). Overall, ExKLoP serves as a robust evaluation platform that streamlines the selection of effective models for self-correcting systems while clearly delineating the types of errors encountered. The complete implementation, along with all relevant data, is available at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09730v2">Sociodemographic Prompting is Not Yet an Effective Approach for Simulating Subjective Judgments with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Human judgments are inherently subjective and are actively affected by personal traits such as gender and ethnicity. While Large Language Models (LLMs) are widely used to simulate human responses across diverse contexts, their ability to account for demographic differences in subjective tasks remains uncertain. In this study, leveraging the POPQUORN dataset, we evaluate nine popular LLMs on their ability to understand demographic differences in two subjective judgment tasks: politeness and offensiveness. We find that in zero-shot settings, most models' predictions for both tasks align more closely with labels from White participants than those from Asian or Black participants, while only a minor gender bias favoring women appears in the politeness task. Furthermore, sociodemographic prompting does not consistently improve and, in some cases, worsens LLMs' ability to perceive language from specific sub-populations. These findings highlight potential demographic biases in LLMs when performing subjective judgment tasks and underscore the limitations of sociodemographic prompting as a strategy to achieve pluralistic alignment. Code and data are available at: https://github.com/Jiaxin-Pei/LLM-as-Subjective-Judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12217v1">Optimal Brain Iterative Merging: Mitigating Interference in LLM Merging</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities, but their high computational costs pose challenges for customization. Model merging offers a cost-effective alternative, yet existing methods suffer from interference among parameters, leading to performance degradation. In this work, we propose Optimal Brain Iterative Merging (OBIM), a novel method designed to mitigate both intra-model and inter-model interference. OBIM consists of two key components: (1) A saliency measurement mechanism that evaluates parameter importance based on loss changes induced by individual weight alterations, reducing intra-model interference by preserving only high-saliency parameters. (2) A mutually exclusive iterative merging framework, which incrementally integrates models using a binary mask to avoid direct parameter averaging, thereby mitigating inter-model interference. We validate OBIM through experiments on both Supervised Fine-Tuned (SFT) models and post-pretrained checkpoints. The results show that OBIM significantly outperforms existing merging techniques. Overall, OBIM provides an effective and practical solution for enhancing LLM merging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12216v1">Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Long-context models are essential for many applications but face inefficiencies in loading large KV caches during decoding. Prior methods enforce fixed token budgets for sparse attention, assuming a set number of tokens can approximate full attention. However, these methods overlook variations in the importance of attention across heads, layers, and contexts. To address these limitations, we propose Tactic, a sparsity-adaptive and calibration-free sparse attention mechanism that dynamically selects tokens based on their cumulative attention scores rather than a fixed token budget. By setting a target fraction of total attention scores, Tactic ensures that token selection naturally adapts to variations in attention sparsity. To efficiently approximate this selection, Tactic leverages clustering-based sorting and distribution fitting, allowing it to accurately estimate token importance with minimal computational overhead. We show that Tactic outperforms existing sparse attention algorithms, achieving superior accuracy and up to 7.29x decode attention speedup. This improvement translates to an overall 1.58x end-to-end inference speedup, making Tactic a practical and effective solution for long-context LLM inference in accuracy-sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12214v1">Zero Token-Driven Deep Thinking in LLMs: Unlocking the Full Potential of Existing Parameters via Cyclic Refinement</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Resource limitations often constrain the parameter counts of Large Language Models (LLMs), hindering their performance. While existing methods employ parameter sharing to reuse the same parameter set under fixed budgets, such approaches typically force each layer to assume multiple roles with a predetermined number of iterations, restricting efficiency and adaptability. In this work, we propose the Zero Token Transformer (ZTT), which features a head-tail decoupled parameter cycling method. We disentangle the first (head) and last (tail) layers from parameter cycling and iteratively refine only the intermediate layers. Furthermore, we introduce a Zero-Token Mechanism, an internal architectural component rather than an input token, to guide layer-specific computation. At each cycle, the model retrieves a zero token (with trainable key values) from a Zero-Token Pool, integrating it alongside regular tokens in the attention mechanism. The corresponding attention scores not only reflect each layer's computational importance but also enable dynamic early exits without sacrificing overall model accuracy. Our approach achieves superior performance under tight parameter budgets, effectively reduces computational overhead via early exits, and can be readily applied to fine-tune existing pre-trained models for enhanced efficiency and adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12134v1">SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) reasoning enables Large Language Models (LLMs) to solve complex reasoning tasks by generating intermediate reasoning steps. However, most existing approaches focus on hard token decoding, which constrains reasoning within the discrete vocabulary space and may not always be optimal. While recent efforts explore continuous-space reasoning, they often suffer from catastrophic forgetting, limiting their applicability to state-of-the-art LLMs that already perform well in zero-shot settings with a proper instruction. To address this challenge, we propose a novel approach for continuous-space reasoning that does not require modifying the underlying LLM. Specifically, we employ a lightweight assistant model to generate instance-specific soft thought tokens speculatively as the initial chain of thoughts, which are then mapped into the LLM's representation space via a projection module. Experimental results on five reasoning benchmarks demonstrate that our method enhances LLM reasoning performance through supervised, parameter-efficient fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09606v2">Human-LLM Coevolution: Evidence from Academic Writing</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      With a statistical analysis of arXiv paper abstracts, we report a marked drop in the frequency of several words previously identified as overused by ChatGPT, such as "delve", starting soon after they were pointed out in early 2024. The frequency of certain other words favored by ChatGPT, such as "significant", has instead kept increasing. These phenomena suggest that some authors of academic papers have adapted their use of large language models (LLMs), for example, by selecting outputs or applying modifications to the LLM-generated content. Such coevolution and cooperation of humans and LLMs thus introduce additional challenges to the detection of machine-generated text in real-world scenarios. Estimating the impact of LLMs on academic writing by examining word frequency remains feasible, and more attention should be paid to words that were already frequently employed, including those that have decreased in frequency due to LLMs' disfavor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12120v1">LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Scaling laws guide the development of large language models (LLMs) by offering estimates for the optimal balance of model size, tokens, and compute. More recently, loss-to-loss scaling laws that relate losses across pretraining datasets and downstream tasks have emerged as a powerful tool for understanding and improving LLM performance. In this work, we investigate which factors most strongly influence loss-to-loss scaling. Our experiments reveal that the pretraining data and tokenizer determine the scaling trend. In contrast, model size, optimization hyperparameters, and even significant architectural differences, such as between transformer-based models like Llama and state-space models like Mamba, have limited impact. Consequently, practitioners should carefully curate suitable pretraining datasets for optimal downstream performance, while architectures and other settings can be freely optimized for training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v1">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code is available at https://github.com/WujiangXu/AgenticMemory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03823v2">Both Text and Images Leaked! A Systematic Analysis of Multimodal LLM Data Contamination</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Code Available: https://github.com/MLLM-Data-Contamination/MM-Detect
    </div>
    <details class="paper-abstract">
      The rapid progression of multimodal large language models (MLLMs) has demonstrated superior performance on various multimodal benchmarks. However, the issue of data contamination during training creates challenges in performance evaluation and comparison. While numerous methods exist for detecting models' contamination in large language models (LLMs), they are less effective for MLLMs due to their various modalities and multiple training phases. In this study, we introduce a multimodal data contamination detection framework, MM-Detect, designed for MLLMs. Our experimental results indicate that MM-Detect is quite effective and sensitive in identifying varying degrees of contamination, and can highlight significant performance improvements due to the leakage of multimodal benchmark training sets. Furthermore, we explore whether the contamination originates from the base LLMs used by MLLMs or the multimodal training phase, providing new insights into the stages at which contamination may be introduced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16491v2">BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      In this work, we tackle the challenge of embedding realistic human personality traits into LLMs. Previous approaches have primarily focused on prompt-based methods that describe the behavior associated with the desired personality traits, suffering from realism and validity issues. To address these limitations, we introduce BIG5-CHAT, a large-scale dataset containing 100,000 dialogues designed to ground models in how humans express their personality in language. Leveraging this dataset, we explore Supervised Fine-Tuning and Direct Preference Optimization as training-based methods to align LLMs more naturally with human personality patterns. Our methods outperform prompting on personality assessments such as BFI and IPIP-NEO, with trait correlations more closely matching human data. Furthermore, our experiments reveal that models trained to exhibit higher conscientiousness, higher agreeableness, lower extraversion, and lower neuroticism display better performance on reasoning tasks, aligning with psychological findings on how these traits impact human cognitive performance. To our knowledge, this work is the first comprehensive study to demonstrate how training-based methods can shape LLM personalities through learning from real human behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12073v1">Can LLMs Simulate Social Media Engagement? A Study on Action-Guided Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Social media enables dynamic user engagement with trending topics, and recent research has explored the potential of large language models (LLMs) for response generation. While some studies investigate LLMs as agents for simulating user behavior on social media, their focus remains on practical viability and scalability rather than a deeper understanding of how well LLM aligns with human behavior. This paper analyzes LLMs' ability to simulate social media engagement through action guided response generation, where a model first predicts a user's most likely engagement action-retweet, quote, or rewrite-towards a trending post before generating a personalized response conditioned on the predicted action. We benchmark GPT-4o-mini, O1-mini, and DeepSeek-R1 in social media engagement simulation regarding a major societal event discussed on X. Our findings reveal that zero-shot LLMs underperform BERT in action prediction, while few-shot prompting initially degrades the prediction accuracy of LLMs with limited examples. However, in response generation, few-shot LLMs achieve stronger semantic alignment with ground truth posts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12067v1">TokenSkip: Controllable Chain-of-Thought Compression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12065v1">Formalizing Complex Mathematical Statements with LLMs: A Study on Mathematical Definitions</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Thanks to their linguistic capabilities, LLMs offer an opportunity to bridge the gap between informal mathematics and formal languages through autoformalization. However, it is still unclear how well LLMs generalize to sophisticated and naturally occurring mathematical statements. To address this gap, we investigate the task of autoformalizing real-world mathematical definitions -- a critical component of mathematical discourse. Specifically, we introduce two novel resources for autoformalisation, collecting definitions from Wikipedia (Def_Wiki) and arXiv papers (Def_ArXiv). We then systematically evaluate a range of LLMs, analyzing their ability to formalize definitions into Isabelle/HOL. Furthermore, we investigate strategies to enhance LLMs' performance including refinement through external feedback from Proof Assistants, and formal definition grounding, where we guide LLMs through relevant contextual elements from formal mathematical libraries. Our findings reveal that definitions present a greater challenge compared to existing benchmarks, such as miniF2F. In particular, we found that LLMs still struggle with self-correction, and aligning with relevant mathematical libraries. At the same time, structured refinement methods and definition grounding strategies yield notable improvements of up to 16% on self-correction capabilities and 43% on the reduction of undefined errors, highlighting promising directions for enhancing LLM-based autoformalization in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12055v1">Designing Role Vectors to Improve LLM Inference Behaviour</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Submitted to ARR 2025 February cycle
    </div>
    <details class="paper-abstract">
      The influence of personas on Large Language Models (LLMs) has been widely studied, yet their direct impact on performance remains uncertain. This work explores a novel approach to guiding LLM behaviour through role vectors, an alternative to persona-based prompting. We construct 29 role vectors derived from model activations and evaluate their impact on benchmark performance across multiple domains. Our analysis investigates whether these vectors can effectively steer models toward domain-specific expertise. We measure two key interventions: (i) activation addition, which reinforces role-specific directions, and (ii) directional ablation, which removes them. Results on well-established benchmarks indicate that role vectors do, in fact, influence model behaviour, improving task performance in relevant domains while marginally affecting unrelated tasks. This, in turn, suggests that manipulating internal model representations has a greater impact on outcomes than persona-based prompting.
    </details>
</div>
