# llm - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13299v1">Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Accepted in NAACL 2025
    </div>
    <details class="paper-abstract">
      Materials discovery and design are essential for advancing technology across various industries by enabling the development of application-specific materials. Recent research has leveraged Large Language Models (LLMs) to accelerate this process. We explore the potential of LLMs to generate viable hypotheses that, once validated, can expedite materials discovery. Collaborating with materials science experts, we curated a novel dataset from recent journal publications, featuring real-world goals, constraints, and methods for designing real-world applications. Using this dataset, we test LLM-based agents that generate hypotheses for achieving given goals under specific constraints. To assess the relevance and quality of these hypotheses, we propose a novel scalable evaluation metric that emulates the process a materials scientist would use to evaluate a hypothesis critically. Our curated dataset, proposed method, and evaluation framework aim to advance future research in accelerating materials discovery and design with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02076v7">LongGenBench: Benchmarking Long-Form Generation in Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 ICLR 2025; Github: https://github.com/mozhu621/LongGenBench/
    </div>
    <details class="paper-abstract">
      Current benchmarks like Needle-in-a-Haystack (NIAH), Ruler, and Needlebench focus on models' ability to understand long-context input sequences but fail to capture a critical dimension: the generation of high-quality long-form text. Applications such as design proposals, technical documentation, and creative writing rely on coherent, instruction-following outputs over extended sequences - a challenge that existing benchmarks do not adequately address. To fill this gap, we introduce LongGenBench, a novel benchmark designed to rigorously evaluate large language models' (LLMs) ability to generate long text while adhering to complex instructions. Through tasks requiring specific events or constraints within generated text, LongGenBench evaluates model performance across four distinct scenarios, three instruction types, and two generation-lengths (16K and 32K tokens). Our evaluation of ten state-of-the-art LLMs reveals that, despite strong results on Ruler, all models struggled with long text generation on LongGenBench, particularly as text length increased. This suggests that current LLMs are not yet equipped to meet the demands of real-world, long-form text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13080v1">Refining Input Guardrails: Enhancing LLM-as-a-Judge Efficiency Through Chain-of-Thought Fine-Tuning and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 16 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated powerful capabilities that render them valuable in different applications, including conversational AI products. It is paramount to ensure the security and reliability of these products by mitigating their vulnerabilities towards malicious user interactions, which can lead to the exposure of great risks and reputational repercussions. In this work, we present a comprehensive study on the efficacy of fine-tuning and aligning Chain-of-Thought (CoT) responses of different LLMs that serve as input moderation guardrails. We systematically explore various tuning methods by leveraging a small set of training data to adapt these models as proxy defense mechanisms to detect malicious inputs and provide a reasoning for their verdicts, thereby preventing the exploitation of conversational agents. We rigorously evaluate the efficacy and robustness of different tuning strategies to generalize across diverse adversarial and malicious query types. Our experimental results outline the potential of alignment processes tailored to a varied range of harmful input queries, even with constrained data resources. These techniques significantly enhance the safety of conversational AI systems and provide a feasible framework for deploying more secure and trustworthy AI-driven interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12980v1">Implicit Causality-biases in humans and LLMs as a tool for benchmarking LLM discourse capabilities</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 38 pages, 8 figures
    </div>
    <details class="paper-abstract">
      In this paper, we compare data generated with mono- and multilingual LLMs spanning a range of model sizes with data provided by human participants in an experimental setting investigating well-established discourse biases. Beyond the comparison as such, we aim to develop a benchmark to assess the capabilities of LLMs with discourse biases as a robust proxy for more general discourse understanding capabilities. More specifically, we investigated Implicit Causality verbs, for which psycholinguistic research has found participants to display biases with regard to three phenomena:\ the establishment of (i) coreference relations (Experiment 1), (ii) coherence relations (Experiment 2), and (iii) the use of particular referring expressions (Experiments 3 and 4). With regard to coreference biases we found only the largest monolingual LLM (German Bloom 6.4B) to display more human-like biases. For coherence relation, no LLM displayed the explanation bias usually found for humans. For referring expressions, all LLMs displayed a preference for referring to subject arguments with simpler forms than to objects. However, no bias effect on referring expression was found, as opposed to recent studies investigating human biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12972v1">Accessible Smart Contracts Verification: Synthesizing Formal Models with Tamed LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      When blockchain systems are said to be trustless, what this really means is that all the trust is put into software. Thus, there are strong incentives to ensure blockchain software is correct -- vulnerabilities here cost millions and break businesses. One of the most powerful ways of establishing software correctness is by using formal methods. Approaches based on formal methods, however, induce a significant overhead in terms of time and expertise required to successfully employ them. Our work addresses this critical disadvantage by automating the creation of a formal model -- a mathematical abstraction of the software system -- which is often a core task when employing formal methods. We perform model synthesis in three phases: we first transpile the code into model stubs; then we "fill in the blanks" using a large language model (LLM); finally, we iteratively repair the generated model, on both syntactical and semantical level. In this way, we significantly reduce the amount of time necessary to create formal models and increase accessibility of valuable software verification methods that rely on them. The practical context of our work was reducing the time-to-value of using formal models for correctness audits of smart contracts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11935v2">Web vs. LLMs: An Empirical Study of Learning Behaviors of CS2 Students</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      LLMs such as ChatGPT have been widely adopted by students in higher education as tools for learning programming and related concepts. However, it remains unclear how effective students are and what strategies students use while learning with LLMs. Since the majority of students' experiences in online self-learning have come through using search engines such as Google, evaluating AI tools in this context can help us address these gaps. In this mixed methods research, we conducted an exploratory within-subjects study to understand how CS2 students learn programming concepts using both LLMs as well as traditional online methods such as educational websites and videos to examine how students approach learning within and across both scenarios. We discovered that students found it easier to learn a more difficult concept using traditional methods than using ChatGPT. We also found that students ask fewer follow-ups and use more keyword-based queries for search engines while their prompts to LLMs tend to explicitly ask for information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12862v1">Mutation-Guided LLM-based Test Generation at Meta</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 Submitted to FSE 2025 Industry Track
    </div>
    <details class="paper-abstract">
      This paper describes Meta's ACH system for mutation-guided LLM-based test generation. ACH generates relatively few mutants (aka simulated faults), compared to traditional mutation testing. Instead, it focuses on generating currently undetected faults that are specific to an issue of concern. From these currently uncaught faults, ACH generates tests that can catch them, thereby `killing' the mutants and consequently hardening the platform against regressions. We use privacy concerns to illustrate our approach, but ACH can harden code against {\em any} type of regression. In total, ACH was applied to 10,795 Android Kotlin classes in 7 software platforms deployed by Meta, from which it generated 9,095 mutants and 571 privacy-hardening test cases. ACH also deploys an LLM-based equivalent mutant detection agent that achieves a precision of 0.79 and a recall of 0.47 (rising to 0.95 and 0.96 with simple pre-processing). ACH was used by Messenger and WhatsApp test-a-thons where engineers accepted 73% of its tests, judging 36% to privacy relevant. We conclude that ACH hardens code against specific concerns and that, even when its tests do not directly tackle the specific concern, engineers find them useful for their other benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12826v1">Open or Closed LLM for Lesser-Resourced Languages? Lessons from Greek</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 NLP, Modern Greek, benchmark, machine learning, language resources
    </div>
    <details class="paper-abstract">
      Natural Language Processing (NLP) for lesser-resourced languages faces persistent challenges, including limited datasets, inherited biases from high-resource languages, and the need for domain-specific solutions. This study addresses these gaps for Modern Greek through three key contributions. First, we evaluate the performance of open-source (Llama-70b) and closed-source (GPT-4o mini) large language models (LLMs) on seven core NLP tasks with dataset availability, revealing task-specific strengths, weaknesses, and parity in their performance. Second, we expand the scope of Greek NLP by reframing Authorship Attribution as a tool to assess potential data usage by LLMs in pre-training, with high 0-shot accuracy suggesting ethical implications for data provenance. Third, we showcase a legal NLP case study, where a Summarize, Translate, and Embed (STE) methodology outperforms the traditional TF-IDF approach for clustering \emph{long} legal texts. Together, these contributions provide a roadmap to advance NLP in lesser-resourced languages, bridging gaps in model evaluation, task innovation, and real-world impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12774v1">LLMs as Repositories of Factual Knowledge: Limitations and Solutions</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      LLMs' sources of knowledge are data snapshots containing factual information about entities collected at different timestamps and from different media types (e.g. wikis, social media, etc.). Such unstructured knowledge is subject to change due to updates through time from past to present. Equally important are the inconsistencies and inaccuracies occurring in different information sources. Consequently, the model's knowledge about an entity may be perturbed while training over the sequence of snapshots or at inference time, resulting in inconsistent and inaccurate model performance. In this work, we study the appropriateness of Large Language Models (LLMs) as repositories of factual knowledge. We consider twenty-four state-of-the-art LLMs that are either closed-, partially (weights), or fully (weight and training data) open-source. We evaluate their reliability in responding to time-sensitive factual questions in terms of accuracy and consistency when prompts are perturbed. We further evaluate the effectiveness of state-of-the-art methods to improve LLMs' accuracy and consistency. We then propose "ENtity-Aware Fine-tuning" (ENAF), a soft neurosymbolic approach aimed at providing a structured representation of entities during fine-tuning to improve the model's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12702v1">Paradigm-Based Automatic HDL Code Generation Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 accepted by ISQED2025. arXiv admin note: text overlap with arXiv:2407.18326
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated the ability to generate hardware description language (HDL) code for digital circuits, they still face the hallucination problem, which can result in the generation of incorrect HDL code or misinterpretation of specifications. In this work, we introduce a human-expert-inspired method to mitigate the hallucination of LLMs and enhance their performance in HDL code generation. We begin by constructing specialized paradigm blocks that consist of several steps designed to divide and conquer generation tasks, mirroring the design methodology of human experts. These steps include information extraction, human-like design flows, and the integration of external tools. LLMs are then instructed to classify the type of circuit in order to match it with the appropriate paradigm block and execute the block to generate the HDL codes. Additionally, we propose a two-phase workflow for multi-round generation, aimed at effectively improving the testbench pass rate of the generated HDL codes within a limited number of generation and verification rounds. Experimental results demonstrate that our method significantly enhances the functional correctness of the generated Verilog code
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12697v1">Combining Knowledge Graph and LLMs for Enhanced Zero-shot Visual Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      Zero-shot visual question answering (ZS-VQA), an emerged critical research area, intends to answer visual questions without providing training samples. Existing research in ZS-VQA has proposed to leverage knowledge graphs or large language models (LLMs), respectively, as external information sources to help VQA model comprehend images and questions. However, LLMs often struggle in accurately interpreting specific question meanings. Meanwhile, although knowledge graph has rich entity relationships, it is challenging to effectively connect entities to individual image content for visual question answers. In this paper, we propose a novel design to combine knowledge graph and LLMs for zero-shot visual question answer. Our approach uses LLMs' powerful understanding capabilities to accurately interpret image content through a strategic question search mechanism. Meanwhile, the knowledge graph is used to expand and connect users' queries to the image content for better visual question answering. An optimization algorithm is further used to determine the optimal weights for the loss functions derived from different information sources, towards a globally optimal set of candidate answers. Experimental results on two benchmark datasets demonstrate that our model achieves state-of-the-art (SOTA) performance. Both source code and benchmark data will be released for public access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07098v2">A Multi-Agent Approach for REST API Testing with Semantic Graphs and LLM-Driven Inputs</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 To be published in the 47th IEEE/ACM International Conference on Software Engineering (ICSE 2025)
    </div>
    <details class="paper-abstract">
      As modern web services increasingly rely on REST APIs, their thorough testing has become crucial. Furthermore, the advent of REST API documentation languages, such as the OpenAPI Specification, has led to the emergence of many black-box REST API testing tools. However, these tools often focus on individual test elements in isolation (e.g., APIs, parameters, values), resulting in lower coverage and less effectiveness in fault detection. To address these limitations, we present AutoRestTest, the first black-box tool to adopt a dependency-embedded multi-agent approach for REST API testing that integrates multi-agent reinforcement learning (MARL) with a semantic property dependency graph (SPDG) and Large Language Models (LLMs). Our approach treats REST API testing as a separable problem, where four agents -- API, dependency, parameter, and value agents -- collaborate to optimize API exploration. LLMs handle domain-specific value generation, the SPDG model simplifies the search space for dependencies using a similarity score between API operations, and MARL dynamically optimizes the agents' behavior. Our evaluation of AutoRestTest on 12 real-world REST services shows that it outperforms the four leading black-box REST API testing tools, including those assisted by RESTGPT (which generates realistic test inputs using LLMs), in terms of code coverage, operation coverage, and fault detection. Notably, AutoRestTest is the only tool able to trigger an internal server error in the Spotify service. Our ablation study illustrates that each component of AutoRestTest -- the SPDG, the LLM, and the agent-learning mechanism -- contributes to its overall effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12599v1">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02406v2">Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly difficult as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. In this paper, we answer the following question: Given a piece of text, can we identify whether it was produced by LLM $A$ or $B$ (where $B$ can be a human)? We model LLM-generated text as a sequential stochastic process with complete dependence on history and design zero-shot statistical tests to distinguish between (i) the text generated by two different sets of LLMs $A$ (in-house) and $B$ (non-sanctioned) and also (ii) LLM-generated and human-generated texts. We prove that the type I and type II errors for our tests decrease exponentially in the text length. In designing our tests, we derive concentration inequalities on the difference between log-perplexity and the average entropy of the string under $A$. Specifically, for a given string, we demonstrate that if the string is generated by $A$, the log-perplexity of the string under $A$ converges to the average entropy of the string under $A$, except with an exponentially small probability in string length. We also show that if $B$ generates the text, except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. Lastly, we present preliminary experimental results to support our theoretical results. By enabling guaranteed (with high probability) finding of the origin of harmful LLM-generated text with arbitrary size, we can help combat misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15023v2">PaperWave: Listening to Research Papers as Conversational Podcasts Scripted by LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      Listening to audio content, such as podcasts and audiobooks, is one way for people to engage with knowledge. Listening affords people more mobility than reading by seeing, thereby broadening their learning opportunities. This study explores the potential applications of large language models (LLMs) to adapt text documents to audio content and addresses the lack of listening-friendly materials for niche content, such as research papers. LLMs can generate scripts of audio content in various styles tailored to specific needs, such as full-content duration or speech types (monologue or dialogue). To explore this potential, we developed PaperWave as a prototype that transforms academic paper PDFs into conversational podcasts. Our two-month investigation, involving 11 participants (including the authors), employed an autobiographical design, a field study, and a design workshop. The findings highlight the importance of considering listener interaction with their environment when designing document-to-audio systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12573v1">Leveraging LLMs to Create a Haptic Devices' Recommendation System</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      Haptic technology has seen significant growth, yet a lack of awareness of existing haptic device design knowledge hinders development. This paper addresses these limitations by leveraging advancements in Large Language Models (LLMs) to develop a haptic agent, focusing specifically on Grounded Force Feedback (GFF) devices recommendation. Our approach involves automating the creation of a structured haptic device database using information from research papers and product specifications. This database enables the recommendation of relevant GFF devices based on user queries. To ensure precise and contextually relevant recommendations, the system employs a dynamic retrieval method that combines both conditional and semantic searches. Benchmarking against the established UEQ and existing haptic device searching tools, the proposed haptic recommendation agent ranks in the top 10\% across all UEQ categories with mean differences favoring the agent in nearly all subscales, and maintains no significant performance bias across different user groups, showcasing superior usability and user satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09879v2">Testing Refactoring Engine via Historical Bug Report driven LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 Accepted at the 2nd ACM international conference on AI Foundation Models and Software Engineering (FORGE 2025)
    </div>
    <details class="paper-abstract">
      Refactoring is the process of restructuring existing code without changing its external behavior while improving its internal structure. Refactoring engines are integral components of modern Integrated Development Environments (IDEs) and can automate or semi-automate this process to enhance code readability, reduce complexity, and improve the maintainability of software products. Similar to traditional software systems such as compilers, refactoring engines may also contain bugs that can lead to unexpected behaviors. In this paper, we propose a novel approach called RETESTER, a LLM-based framework for automated refactoring engine testing. Specifically, by using input program structure templates extracted from historical bug reports and input program characteristics that are error-prone, we design chain-of-thought (CoT) prompts to perform refactoring-preserving transformations. The generated variants are then tested on the latest version of refactoring engines using differential testing. We evaluate RETESTER on two most popular modern refactoring engines (i.e., ECLIPSE, and INTELLIJ IDEA). It successfully revealed 18 new bugs in the latest version of those refactoring engines. By the time we submit our paper, seven of them were confirmed by their developers, and three were fixed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12557v1">Understanding the LLM-ification of CHI: Unpacking the Impact of LLMs at CHI through a Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 This is a preprint version of the paper conditionally accepted to CHI'25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been positioned to revolutionize HCI, by reshaping not only the interfaces, design patterns, and sociotechnical systems that we study, but also the research practices we use. To-date, however, there has been little understanding of LLMs' uptake in HCI. We address this gap via a systematic literature review of 153 CHI papers from 2020-24 that engage with LLMs. We taxonomize: (1) domains where LLMs are applied; (2) roles of LLMs in HCI projects; (3) contribution types; and (4) acknowledged limitations and risks. We find LLM work in 10 diverse domains, primarily via empirical and artifact contributions. Authors use LLMs in five distinct roles, including as research tools or simulated users. Still, authors often raise validity and reproducibility concerns, and overwhelmingly study closed models. We outline opportunities to improve HCI research with and on LLMs, and provide guiding questions for researchers to consider the validity and appropriateness of LLM-related work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15524v2">The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive potential to simulate human behavior. We identify a fundamental challenge in using them to simulate experiments: when LLM-simulated subjects are blind to the experimental design (as is standard practice with human subjects), variations in treatment systematically affect unspecified variables that should remain constant, violating the unconfoundedness assumption. Using demand estimation as a context and an actual experiment as a benchmark, we show this can lead to implausible results. While confounding may in principle be addressed by controlling for covariates, this can compromise ecological validity in the context of LLM simulations: controlled covariates become artificially salient in the simulated decision process, which introduces focalism. This trade-off between unconfoundedness and ecological validity is usually absent in traditional experimental design and represents a unique challenge in LLM simulations. We formalize this challenge theoretically, showing it stems from ambiguous prompting strategies, and hence cannot be fully addressed by improving training data or by fine-tuning. Alternative approaches that unblind the experimental design to the LLM show promise. Our findings suggest that effectively leveraging LLMs for experimental simulations requires fundamentally rethinking established experimental design practices rather than simply adapting protocols developed for human subjects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21548v2">MultiTok: Variable-Length Tokenization for Efficient LLMs Adapted from LZW Compression</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large language models have drastically changed the prospects of AI by introducing technologies for more complex natural language processing. However, current methodologies to train such LLMs require extensive resources including but not limited to large amounts of data, expensive machinery, and lengthy training. To solve this problem, this paper proposes a new tokenization method inspired by universal Lempel-Ziv-Welch data compression that compresses repetitive phrases into multi-word tokens. With MultiTok as a new tokenizing tool, we show that language models are able to be trained notably more efficiently while offering a similar accuracy on more succinct and compressed training data. In fact, our results demonstrate that MultiTok achieves a comparable performance to the BERT and GPT-2 standards as both a stand-alone tokenizer and an add-on to existing tokenizers while also providing close to 2.5x faster training with more than 30% less training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12456v1">Deploying Privacy Guardrails for LLMs: A Comparative Analysis of Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 This paper has been accepted at Deployable AI workshop at AAAI 2025
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) has revolutionized AI applications but poses significant challenges in safeguarding user privacy. Ensuring compliance with privacy regulations such as GDPR and CCPA while addressing nuanced privacy risks requires robust and scalable frameworks. This paper presents a detailed study of OneShield Privacy Guard, a framework designed to mitigate privacy risks in user inputs and LLM outputs across enterprise and open-source settings. We analyze two real-world deployments:(1) a multilingual privacy-preserving system integrated with Data and Model Factory, focusing on enterprise-scale data governance; and (2) PR Insights, an open-source repository emphasizing automated triaging and community-driven refinements. In Deployment 1, OneShield achieved a 0.95 F1 score in detecting sensitive entities like dates, names, and phone numbers across 26 languages, outperforming state-of-the-art tool such as StarPII and Presidio by up to 12\%. Deployment 2, with an average F1 score of 0.86, reduced manual effort by over 300 hours in three months, accurately flagging 8.25\% of 1,256 pull requests for privacy risks with enhanced context sensitivity. These results demonstrate OneShield's adaptability and efficacy in diverse environments, offering actionable insights for context-aware entity recognition, automated compliance, and ethical AI adoption. This work advances privacy-preserving frameworks, supporting user trust and compliance across operational contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v1">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 14 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve a strong performance with 67.41\% on BIRD benchmark (dev) without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12332v1">Automatic Labelling with Open-source LLMs using Dynamic Label Schema Integration</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 11 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Acquiring labelled training data remains a costly task in real world machine learning projects to meet quantity and quality requirements. Recently Large Language Models (LLMs), notably GPT-4, have shown great promises in labelling data with high accuracy. However, privacy and cost concerns prevent the ubiquitous use of GPT-4. In this work, we explore effectively leveraging open-source models for automatic labelling. We identify integrating label schema as a promising technology but found that naively using the label description for classification leads to poor performance on high cardinality tasks. To address this, we propose Retrieval Augmented Classification (RAC) for which LLM performs inferences for one label at a time using corresponding label schema; we start with the most related label and iterates until a label is chosen by the LLM. We show that our method, which dynamically integrates label description, leads to performance improvements in labelling tasks. We further show that by focusing only on the most promising labels, RAC can trade off between label quality and coverage - a property we leverage to automatically label our internal datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10622v3">A recent evaluation on the performance of LLMs on radiation oncology physics using questions of randomly shuffled options</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Purpose: We present an updated study evaluating the performance of large language models (LLMs) in answering radiation oncology physics questions, focusing on the recently released models. Methods: A set of 100 multiple-choice radiation oncology physics questions, previously created by a well-experienced physicist, was used for this study. The answer options of the questions were randomly shuffled to create "new" exam sets. Five LLMs -- OpenAI o1-preview, GPT-4o, LLaMA 3.1 (405B), Gemini 1.5 Pro, and Claude 3.5 Sonnet -- with the versions released before September 30, 2024, were queried using these new exam sets. To evaluate their deductive reasoning ability, the correct answer options in the questions were replaced with "None of the above." Then, the explain-first and step-by-step instruction prompts were used to test if this strategy improved their reasoning ability. The performance of the LLMs was compared with the answers from medical physicists. Results: All models demonstrated expert-level performance on these questions, with o1-preview even surpassing medical physicists with a majority vote. When replacing the correct answer options with 'None of the above', all models exhibited a considerable decline in performance, suggesting room for improvement. The explain-first and step-by-step instruction prompts helped enhance the reasoning ability of the LLaMA 3.1 (405B), Gemini 1.5 Pro, and Claude 3.5 Sonnet models. Conclusion: These recently released LLMs demonstrated expert-level performance in answering radiation oncology physics questions, exhibiting great potential to assist in radiation oncology physics education and training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12300v1">LLM-Assisted Knowledge Graph Completion for Curriculum and Domain Modelling in Personalized Higher Education Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted in the IEEE Global Engineering Education Conference (EDUCON2025), London, UK, 22-25 April, 2025
    </div>
    <details class="paper-abstract">
      While learning personalization offers great potential for learners, modern practices in higher education require a deeper consideration of domain models and learning contexts, to develop effective personalization algorithms. This paper introduces an innovative approach to higher education curriculum modelling that utilizes large language models (LLMs) for knowledge graph (KG) completion, with the goal of creating personalized learning-path recommendations. Our research focuses on modelling university subjects and linking their topics to corresponding domain models, enabling the integration of learning modules from different faculties and institutions in the student's learning path. Central to our approach is a collaborative process, where LLMs assist human experts in extracting high-quality, fine-grained topics from lecture materials. We develop a domain, curriculum, and user models for university modules and stakeholders. We implement this model to create the KG from two study modules: Embedded Systems and Development of Embedded Systems Using FPGA. The resulting KG structures the curriculum and links it to the domain models. We evaluate our approach through qualitative expert feedback and quantitative graph quality metrics. Domain experts validated the relevance and accuracy of the model, while the graph quality metrics measured the structural properties of our KG. Our results show that the LLM-assisted graph completion approach enhances the ability to connect related courses across disciplines to personalize the learning experience. Expert feedback also showed high acceptance of the proposed collaborative approach for concept extraction and classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12273v1">Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Tech Report. Github: https://github.com/InternLM/Condor
    </div>
    <details class="paper-abstract">
      The quality of Supervised Fine-Tuning (SFT) data plays a critical role in enhancing the conversational capabilities of Large Language Models (LLMs). However, as LLMs become more advanced, the availability of high-quality human-annotated SFT data has become a significant bottleneck, necessitating a greater reliance on synthetic training data. In this work, we introduce Condor, a novel two-stage synthetic data generation framework that incorporates World Knowledge Tree and Self-Reflection Refinement to produce high-quality SFT data at scale. Our experimental results demonstrate that a base model fine-tuned on only 20K Condor-generated samples achieves superior performance compared to counterparts. The additional refinement stage in Condor further enables iterative self-improvement for LLMs at various scales (up to 72B), validating the effectiveness of our approach. Furthermore, our investigation into the scaling for synthetic data in post-training reveals substantial unexplored potential for performance improvements, opening promising avenues for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12210v1">You Can't Eat Your Cake and Have It Too: The Performance Degradation of LLMs with Jailbreak Defense</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      With the rise of generative large language models (LLMs) like LLaMA and ChatGPT, these models have significantly transformed daily life and work by providing advanced insights. However, as jailbreak attacks continue to circumvent built-in safety mechanisms, exploiting carefully crafted scenarios or tokens, the safety risks of LLMs have come into focus. While numerous defense strategies--such as prompt detection, modification, and model fine-tuning--have been proposed to counter these attacks, a critical question arises: do these defenses compromise the utility and usability of LLMs for legitimate users? Existing research predominantly focuses on the effectiveness of defense strategies without thoroughly examining their impact on performance, leaving a gap in understanding the trade-offs between LLM safety and performance. Our research addresses this gap by conducting a comprehensive study on the utility degradation, safety elevation, and exaggerated-safety escalation of LLMs with jailbreak defense strategies. We propose USEBench, a novel benchmark designed to evaluate these aspects, along with USEIndex, a comprehensive metric for assessing overall model performance. Through experiments on seven state-of-the-art LLMs, we found that mainstream jailbreak defenses fail to ensure both safety and performance simultaneously. Although model-finetuning performs the best overall, their effectiveness varies across LLMs. Furthermore, vertical comparisons reveal that developers commonly prioritize performance over safety when iterating or fine-tuning their LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12162v1">AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      This paper introduces AdaServe, the first LLM serving system to support SLO customization through fine-grained speculative decoding. AdaServe leverages the logits of a draft model to predict the speculative accuracy of tokens and employs a theoretically optimal algorithm to construct token trees for verification. To accommodate diverse SLO requirements without compromising throughput, AdaServe employs a speculation-and-selection scheme that first constructs candidate token trees for each request and then dynamically selects tokens to meet individual SLO constraints while optimizing throughput. Comprehensive evaluations demonstrate that AdaServe achieves up to 73% higher SLO attainment and 74% higher goodput compared to state-of-the-art systems. These results underscore AdaServe's potential to enhance the efficiency and adaptability of LLM deployments across varied application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12152v1">Contextualizing Recommendation Explanations with LLMs: A User Study</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly prevalent in recommender systems, where LLMs can be used to generate personalized recommendations. Here, we examine how different LLM-generated explanations for movie recommendations affect users' perceptions of cognitive, affective, and utilitarian needs and consumption intentions. In a pre-registered, between-subject online experiment (N=759) and follow-up interviews (N=30), we compare (a) LLM-generated generic explanations, and (b) LLM-generated contextualized explanations. Our findings show that contextualized explanations (i.e., explanations that incorporate users' past behaviors) effectively meet users' cognitive needs while increasing users' intentions to watch recommended movies. However, adding explanations offers limited benefits in meeting users' utilitarian and affective needs, raising concerns about the proper design and implications of LLM-generated explanations. Qualitative insights from interviews reveal that referencing users' past preferences enhances trust and understanding but can feel excessive if overused. Furthermore, users with more active and positive engagement with the recommender system and movie-watching get substantial gains from contextualized explanations. Overall, our research clarifies how LLM-generated recommendations influence users' motivations and behaviors, providing valuable insights for the future development of user-centric recommender systems, a key element in social media platforms and online ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12134v1">Do LLMs Provide Links to Code Similar to what they Generate? A Study with Gemini and Bing CoPilot</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are currently used for various software development tasks, including generating code snippets to solve specific problems. Unlike reuse from the Web, LLMs are limited in providing provenance information about the generated code, which may have important trustworthiness and legal consequences. While LLM-based assistants may provide external links that are "related" to the generated code, we do not know how relevant such links are. This paper presents the findings of an empirical study assessing the extent to which 243 and 194 code snippets, across six programming languages, generated by Bing CoPilot and Google Gemini, likely originate from the links provided by these two LLM-based assistants. The study leverages automated code similarity assessments with thorough manual analysis. The study's findings indicate that the LLM-based assistants provide a mix of relevant and irrelevant links having a different nature. Specifically, although 66% of the links from Bing CoPilot and 28% from Google Gemini are relevant, LLMs-based assistants still suffer from serious "provenance debt".
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12128v1">Evaluating Efficiency and Engagement in Scripted and LLM-Enhanced Human-Robot Interactions</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted as a Late-Breaking Report to the 2025, 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI)
    </div>
    <details class="paper-abstract">
      To achieve natural and intuitive interaction with people, HRI frameworks combine a wide array of methods for human perception, intention communication, human-aware navigation and collaborative action. In practice, when encountering unpredictable behavior of people or unexpected states of the environment, these frameworks may lack the ability to dynamically recognize such states, adapt and recover to resume the interaction. Large Language Models (LLMs), owing to their advanced reasoning capabilities and context retention, present a promising solution for enhancing robot adaptability. This potential, however, may not directly translate to improved interaction metrics. This paper considers a representative interaction with an industrial robot involving approach, instruction, and object manipulation, implemented in two conditions: (1) fully scripted and (2) including LLM-enhanced responses. We use gaze tracking and questionnaires to measure the participants' task efficiency, engagement, and robot perception. The results indicate higher subjective ratings for the LLM condition, but objective metrics show that the scripted condition performs comparably, particularly in efficiency and focus during simple tasks. We also note that the scripted condition may have an edge over LLM-enhanced responses in terms of response latency and energy consumption, especially for trivial and repetitive interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13126v1">Preference Curriculum: LLMs Should Always Be Pretrained on Their Preferred Data</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 18 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Current large language models (LLMs) generally utilize a consistent data distribution throughout the entire pretraining process. However, as the model's ability improves, it intuitively should be pretrained with differentiated data. To achieve it, we propose the Perplexity Difference based Preference Curriculum learning (PDPC) framework, which always perceives and uses the data preferred by LLMs to train and boost them. Firstly, we introduce the PD metric to measure the difference in how well strong and weak models fit the samples. Samples with high PD are more challenging for weak models to learn and are more suitable to be arranged in the later stage of pretraining. Secondly, we propose the PD preference function to approximate the model and predict the data preference of the LLM at any time, so as to complete the arrangement of the entire data offline and ensure continuous training without interruption. Experimental results on 1.3B and 3B models demonstrate that our PDPC significantly surpasses baselines. Notably, the 3B model achieved more substantial gains, with an increased average accuracy of over 4.1% across various benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02044v2">QROA: A Black-Box Query-Response Optimization Attack on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have surged in popularity in recent months, yet they possess concerning capabilities for generating harmful content when manipulated. This study introduces the Query-Response Optimization Attack (QROA), an optimization-based strategy designed to exploit LLMs through a black-box, query-only interaction. QROA adds an optimized trigger to a malicious instruction to compel the LLM to generate harmful content. Unlike previous approaches, QROA does not require access to the model's logit information or any other internal data and operates solely through the standard query-response interface of LLMs. Inspired by deep Q-learning and Greedy coordinate descent, the method iteratively updates tokens to maximize a designed reward function. We tested our method on various LLMs such as Vicuna, Falcon, and Mistral, achieving an Attack Success Rate (ASR) over 80\%. We also tested the model against Llama2-chat, the fine-tuned version of Llama2 designed to resist Jailbreak attacks, achieving good ASR with a suboptimal initial trigger seed. This study demonstrates the feasibility of generating jailbreak attacks against deployed LLMs in the public domain using black-box optimization methods, enabling more comprehensive safety testing of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01144v3">BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      The rapidly increasing size of large language models (LLMs) presents significant challenges in memory usage and computational costs. Quantizing both weights and activations can address these issues, with hardware-supported fine-grained scaling emerging as a promising solution to mitigate outliers. However, existing methods struggle to capture nuanced block data distributions. We propose BlockDialect, a block-wise fine-grained mixed format technique that assigns a per-block optimal number format from a formatbook for better data representation. Additionally, we introduce DialectFP4, a formatbook of FP4 variants (akin to dialects) that adapt to diverse data distributions. To leverage this efficiently, we propose a two-stage approach for online DialectFP4 activation quantization. Importantly, DialectFP4 ensures energy efficiency by selecting representable values as scaled integers compatible with low-precision integer arithmetic. BlockDialect achieves 10.78% (7.48%) accuracy gain on the LLaMA3-8B (LLaMA2-7B) model compared to MXFP4 format with lower bit usage per data, while being only 5.45% (2.69%) below full precision even when quantizing full-path matrix multiplication. Focusing on how to represent over how to scale, our work presents a promising path for energy-efficient LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11911v1">Integrate Temporal Graph Learning into LLM-based Temporal Knowledge Graph Model</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Temporal Knowledge Graph Forecasting (TKGF) aims to predict future events based on the observed events in history. Recently, Large Language Models (LLMs) have exhibited remarkable capabilities, generating significant research interest in their application for reasoning over temporal knowledge graphs (TKGs). Existing LLM-based methods have integrated retrieved historical facts or static graph representations into LLMs. Despite the notable performance of LLM-based methods, they are limited by the insufficient modeling of temporal patterns and ineffective cross-modal alignment between graph and language, hindering the ability of LLMs to fully grasp the temporal and structural information in TKGs. To tackle these issues, we propose a novel framework TGL-LLM to integrate temporal graph learning into LLM-based temporal knowledge graph model. Specifically, we introduce temporal graph learning to capture the temporal and relational patterns and obtain the historical graph embedding. Furthermore, we design a hybrid graph tokenization to sufficiently model the temporal patterns within LLMs. To achieve better alignment between graph and language, we employ a two-stage training paradigm to finetune LLMs on high-quality and diverse data, thereby resulting in better performance. Extensive experiments on three real-world datasets show that our approach outperforms a range of state-of-the-art (SOTA) methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11877v1">From Drafts to Answers: Unlocking LLM Potential via Aggregation Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 20 pages; work in progress
    </div>
    <details class="paper-abstract">
      Scaling data and model size has been proven effective for boosting the performance of large language models. In addition to training-time scaling, recent studies have revealed that increasing test-time computational resources can further improve performance. In this work, we introduce Aggregation Fine-Tuning (AFT), a supervised finetuning paradigm where the model learns to synthesize multiple draft responses, referred to as proposals, into a single, refined answer, termed aggregation. At inference time, a propose-and-aggregate strategy further boosts performance by iteratively generating proposals and aggregating them. Empirical evaluations on benchmark datasets show that AFT-trained models substantially outperform standard SFT. Notably, an AFT model, fine-tuned from Llama3.1-8B-Base with only 64k data, achieves a 41.3% LC win rate on AlpacaEval 2, surpassing significantly larger LLMs such as Llama3.1-405B-Instruct and GPT4. By combining sequential refinement and parallel sampling, the propose-and-aggregate framework scales inference-time computation in a flexible manner. Overall, These findings position AFT as a promising approach to unlocking additional capabilities of LLMs without resorting to increasing data volume or model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12624v5">Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Offering a promising solution to the scalability challenges associated with human evaluation, the LLM-as-a-judge paradigm is rapidly gaining traction as an approach to evaluating large language models (LLMs). However, there are still many open questions about the strengths and weaknesses of this paradigm, and what potential biases it may hold. In this paper, we present a comprehensive study of the performance of various LLMs acting as judges, focusing on a clean scenario in which inter-human agreement is high. Investigating thirteen judge models of different model sizes and families, judging answers of nine different 'examtaker models' - both base and instruction-tuned - we find that only the best (and largest) models achieve reasonable alignment with humans. However, they are still quite far behind inter-human agreement and their assigned scores may still differ with up to 5 points from human-assigned scores. In terms of their ranking of the nine exam-taker models, instead, also smaller models and even the lexical metric contains may provide a reasonable signal. Through error analysis and other studies, we identify vulnerabilities in judge models, such as their sensitivity to prompt complexity and length, and a tendency toward leniency. The fact that even the best judges differ from humans in this comparatively simple setup suggest that caution may be wise when using judges in more complex setups. Lastly, our research rediscovers the importance of using alignment metrics beyond simple percent alignment, showing that judges with high percent agreement can still assign vastly different scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11051v2">FLAME: Learning to Navigate with Multimodal LLM in Urban Environments</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted to AAAI 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated potential in Vision-and-Language Navigation (VLN) tasks, yet current applications face challenges. While LLMs excel in general conversation scenarios, they struggle with specialized navigation tasks, yielding suboptimal performance compared to specialized VLN models. We introduce FLAME (FLAMingo-Architected Embodied Agent), a novel Multimodal LLM-based agent and architecture designed for urban VLN tasks that efficiently handles multiple observations. Our approach implements a three-phase tuning technique for effective adaptation to navigation tasks, including single perception tuning for street view description, multiple perception tuning for route summarization, and end-to-end training on VLN datasets. The augmented datasets are synthesized automatically. Experimental results demonstrate FLAME's superiority over existing methods, surpassing state-of-the-art methods by a 7.3% increase in task completion on Touchdown dataset. This work showcases the potential of Multimodal LLMs (MLLMs) in complex navigation tasks, representing an advancement towards applications of MLLMs in the field of embodied intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11864v1">LLM-Agents Driven Automated Simulation Testing and Analysis of small Uncrewed Aerial Systems</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted as full paper at ICSE-2025
    </div>
    <details class="paper-abstract">
      Thorough simulation testing is crucial for validating the correct behavior of small Uncrewed Aerial Systems (sUAS) across multiple scenarios, including adverse weather conditions (such as wind, and fog), diverse settings (hilly terrain, or urban areas), and varying mission profiles (surveillance, tracking). While various sUAS simulation tools exist to support developers, the entire process of creating, executing, and analyzing simulation tests remains a largely manual and cumbersome task. Developers must identify test scenarios, set up the simulation environment, integrate the System under Test (SuT) with simulation tools, formulate mission plans, and collect and analyze results. These labor-intensive tasks limit the ability of developers to conduct exhaustive testing across a wide range of scenarios. To alleviate this problem, in this paper, we propose AutoSimTest, a Large Language Model (LLM)-driven framework, where multiple LLM agents collaborate to support the sUAS simulation testing process. This includes: (1) creating test scenarios that subject the SuT to unique environmental contexts; (2) preparing the simulation environment as per the test scenario; (3) generating diverse sUAS missions for the SuT to execute; and (4) analyzing simulation results and providing an interactive analytics interface. Further, the design of the framework is flexible for creating and testing scenarios for a variety of sUAS use cases, simulation tools, and SuT input requirements. We evaluated our approach by (a) conducting simulation testing of PX4 and ArduPilot flight-controller-based SuTs, (b) analyzing the performance of each agent, and (c) gathering feedback from sUAS developers. Our findings indicate that AutoSimTest significantly improves the efficiency and scope of the sUAS testing process, allowing for more comprehensive and varied scenario evaluations while reducing the manual effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00253v4">CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based Verification</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted by AAAI 2025 main conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant progress in code generation, offering developers groundbreaking automated programming support. However, LLMs often generate code that is syntactically correct and even semantically plausible, but may not execute as expected or fulfill specified requirements. This phenomenon of hallucinations in the code domain has not been systematically explored. To advance the community's understanding and research on this issue, we introduce the concept of code hallucinations and propose a classification method for code hallucination based on execution verification. We categorize code hallucinations into four main types: mapping, naming, resource, and logic hallucinations, with each category further divided into different subcategories to understand and address the unique challenges faced by LLMs in code generation with finer granularity. Additionally, we present a dynamic detection algorithm called CodeHalu designed to detect and quantify code hallucinations. We also introduce the CodeHaluEval benchmark, which includes 8,883 samples from 699 tasks, to systematically and quantitatively evaluate code hallucinations. By evaluating 17 popular LLMs using this benchmark, we reveal significant differences in their accuracy and reliability in code generation, offering detailed insights for further improving the code generation capabilities of LLMs. The CodeHalu benchmark and code are publicly available at https://github.com/yuchen814/CodeHalu.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08067v3">Reward-Augmented Data Enhances Direct Preference Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Preference alignment in Large Language Models (LLMs) has significantly improved their ability to adhere to human instructions and intentions. However, existing direct alignment algorithms primarily focus on relative preferences and often overlook the qualitative aspects of responses. Striving to maximize the implicit reward gap between the chosen and the slightly inferior rejected responses can cause overfitting and unnecessary unlearning of the high-quality rejected responses. The unawareness of the reward scores also drives the LLM to indiscriminately favor the low-quality chosen responses and fail to generalize to responses with the highest rewards, which are sparse in data. To overcome these shortcomings, our study introduces reward-conditioned LLM policies that discern and learn from the entire spectrum of response quality within the dataset, helping extrapolate to more optimal regions. We propose an effective yet simple data relabeling method that conditions the preference pairs on quality scores to construct a reward-augmented dataset. This dataset is easily integrated with existing direct alignment algorithms and is applicable to any preference dataset. The experimental results across instruction-following benchmarks including AlpacaEval, MT-Bench, and Arena-Hard-Auto demonstrate that our approach consistently boosts the performance of DPO by a considerable margin across diverse models. Additionally, our method improves the average accuracy on various academic benchmarks. When applying our method to on-policy data, the resulting DPO model achieves SOTA results on AlpacaEval. Through ablation studies, we demonstrate that our method not only maximizes the utility of preference data but also mitigates the issue of unlearning, demonstrating its broad effectiveness beyond mere dataset expansion. Our code is available at https://github.com/shenao-zhang/reward-augmented-preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11833v1">Is your LLM trapped in a Mental Set? Investigative study on how mental sets affect the reasoning capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      In this paper, we present an investigative study on how Mental Sets influence the reasoning capabilities of LLMs. LLMs have excelled in diverse natural language processing (NLP) tasks, driven by advancements in parameter-efficient fine-tuning (PEFT) and emergent capabilities like in-context learning (ICL). For complex reasoning tasks, selecting the right model for PEFT or ICL is critical, often relying on scores on benchmarks such as MMLU, MATH, and GSM8K. However, current evaluation methods, based on metrics like F1 Score or reasoning chain assessments by larger models, overlook a key dimension: adaptability to unfamiliar situations and overcoming entrenched thinking patterns. In cognitive psychology, Mental Set refers to the tendency to persist with previously successful strategies, even when they become inefficient - a challenge for problem solving and reasoning. We compare the performance of LLM models like Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct and GPT-4o in the presence of mental sets. To the best of our knowledge, this is the first study to integrate cognitive psychology concepts into the evaluation of LLMs for complex reasoning tasks, providing deeper insights into their adaptability and problem-solving efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11779v1">Glinthawk: A Two-Tiered Architecture for High-Throughput LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) have revolutionized natural language processing, but their inference demands substantial resources, while under-utilizing high-end accelerators like GPUs. A major bottleneck arises from the attention mechanism, which requires storing large key-value caches, limiting the maximum achievable throughput way below the available computing resources. Current approaches attempt to mitigate this issue through memory-efficient attention and paging mechanisms, but remained constrained by the assumption that all operations must be performed on high-end accelerators. In this work, we propose Glinthawk, a two-tiered architecture that decouples the attention mechanism from the rest of the Transformer model. This approach allows the memory requirements for attention to scale independently, enabling larger batch sizes and more efficient use of the high-end accelerators. We prototype Glinthawk with NVIDIA T4 GPUs as one tier and standard CPU VMs as the other. Compared to a traditional single-tier setup, it improves throughput by $5.9\times$ and reduces cost of generation by $2.8\times$. For longer sequence lengths, it achieves $16.3\times$ throughput improvement at $2.4\times$ less cost. Our evaluation shows that this architecture can tolerate moderate network latency with minimal performance degradation, making it highly effective for latency-tolerant, throughput-oriented applications such as batch processing. We shared our prototype publicly at \url{https://github.com/microsoft/glinthawk}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17167v1">QualityFlow: An Agentic Workflow for Program Synthesis Controlled by LLM Quality Checks</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      We introduce QualityFlow, a dynamic agentic workflow for program synthesis. Given the English description of a programming problem and a set of unit tests, the model's goal is to synthesize the correct program that solves the problem and passes the tests. QualityFlow consists of multiple large language model (LLM) agents that resemble a software development team, including code generation, testing, and self-debugging. Existing program synthesis methods face three major limitations: assumption of visible unit test conformity, bottleneck of synthesized test quality, and deviation of self-debugging trajectory. To address them, we propose the LLM Quality Checker, which explicitly "imagines" whether the synthesized programs' execution would conform to the unit tests. The Quality Checks dynamically control the workflow, including actions to submit the final answer, clarify the problem statement, and revert previous workflow steps. As a result, our Quality Checker can precisely accept any correct program, mitigate faulty synthesized tests, and prevent potential workflow deviation. The success of the Quality Checker further enables Diversified Prompting, which encourages variations in LLM responses to maximize the possibility that a correct program appears and passes the quality check. In experiments, QualityFlow establishes the state-of-the-art results on four program synthesis benchmarks: MBPP, HumanEval, and the stricter evaluations of both MBPP and HumanEval from EvalPlus. Our systematic analysis shows that the dynamic workflow controlled by LLM quality checks can outperform static workflows and single-attempt zero-shot synthesis. The Quality Checker is the center of our investigation, and we dissect its individual performance and integrated impact on the workflow accuracy, as well as other ablations experiments to justify our workflow design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11721v1">Explain-Query-Test: Self-Evaluating LLMs Via Explanation and Comprehension Discrepancy</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in generating detailed and coherent explanations of complex concepts. However, the extent to which these models truly comprehend the concepts they articulate remains unclear. To assess the level of comprehension of a model relative to the content it generates, we implemented a self-evaluation pipeline where models: (i) given a topic generate an excerpt with information about the topic, (ii) given an excerpt generate question-answer pairs, and finally (iii) given a question generate an answer. We refer to this self-evaluation approach as Explain-Query-Test (EQT). Interestingly, the accuracy on generated questions resulting from running the EQT pipeline correlates strongly with the model performance as verified by typical benchmarks such as MMLU-Pro. In other words, EQT's performance is predictive of MMLU-Pro's, and EQT can be used to rank models without the need for any external source of evaluation data other than lists of topics of interest. Moreover, our results reveal a disparity between the models' ability to produce detailed explanations and their performance on questions related to those explanations. This gap highlights fundamental limitations in the internal knowledge representation and reasoning abilities of current LLMs. We release the code at https://github.com/asgsaeid/EQT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11709v1">Towards Detecting Prompt Knowledge Gaps for Improved LLM-guided Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential in software development, especially for issue resolution. However, despite their widespread use, significant challenges persist in the quality of LLM responses to issue resolution queries. LLM interactions often yield incorrect, incomplete, or ambiguous information, largely due to knowledge gaps in prompt design, which can lead to unproductive exchanges and reduced developer productivity. In this paper, we analyze 433 developer-ChatGPT conversations within GitHub issue threads to examine the impact of prompt knowledge gaps and conversation styles on issue resolution. We identify four main knowledge gaps in developer prompts: Missing Context, Missing Specifications, Multiple Context, and Unclear Instructions. Assuming that conversations within closed issues contributed to successful resolutions while those in open issues did not, we find that ineffective conversations contain knowledge gaps in 54.7% of prompts, compared to only 13.2% in effective ones. Additionally, we observe seven distinct conversational styles, with Directive Prompting, Chain of Thought, and Responsive Feedback being the most prevalent. We find that knowledge gaps are present in all styles of conversations, with Missing Context being the most repeated challenge developers face in issue-resolution conversations. Based on our analysis, we identify key textual and code related heuristics-Specificity, Contextual Richness, and Clarity-that are associated with successful issue closure and help assess prompt quality. These heuristics lay the foundation for an automated tool that can dynamically flag unclear prompts and suggest structured improvements. To test feasibility, we developed a lightweight browser extension prototype for detecting prompt gaps, that can be easily adapted to other tools within developer workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13120v1">Multilinguality in LLM-Designed Reward Functions for Restless Bandits: Effects on Task Performance and Fairness</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 Accepted at the AAAI-2025 Deployable AI Workshop
    </div>
    <details class="paper-abstract">
      Restless Multi-Armed Bandits (RMABs) have been successfully applied to resource allocation problems in a variety of settings, including public health. With the rapid development of powerful large language models (LLMs), they are increasingly used to design reward functions to better match human preferences. Recent work has shown that LLMs can be used to tailor automated allocation decisions to community needs using language prompts. However, this has been studied primarily for English prompts and with a focus on task performance only. This can be an issue since grassroots workers, especially in developing countries like India, prefer to work in local languages, some of which are low-resource. Further, given the nature of the problem, biases along population groups unintended by the user are also undesirable. In this work, we study the effects on both task performance and fairness when the DLM algorithm, a recent work on using LLMs to design reward functions for RMABs, is prompted with non-English language commands. Specifically, we run the model on a synthetic environment for various prompts translated into multiple languages. The prompts themselves vary in complexity. Our results show that the LLM-proposed reward functions are significantly better when prompted in English compared to other languages. We also find that the exact phrasing of the prompt impacts task performance. Further, as prompt complexity increases, performance worsens for all languages; however, it is more robust with English prompts than with lower-resource languages. On the fairness side, we find that low-resource languages and more complex prompts are both highly likely to create unfairness along unintended dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11623v1">Early evidence of how LLMs outperform traditional systems on OCR/HTR tasks for historical records</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We explore the ability of two LLMs -- GPT-4o and Claude Sonnet 3.5 -- to transcribe historical handwritten documents in a tabular format and compare their performance to traditional OCR/HTR systems: EasyOCR, Keras, Pytesseract, and TrOCR. Considering the tabular form of the data, two types of experiments are executed: one where the images are split line by line and the other where the entire scan is used as input. Based on CER and BLEU, we demonstrate that LLMs outperform the conventional OCR/HTR methods. Moreover, we also compare the evaluated CER and BLEU scores to human evaluations to better judge the outputs of whole-scan experiments and understand influential factors for CER and BLEU. Combining judgments from all the evaluation metrics, we conclude that two-shot GPT-4o for line-by-line images and two-shot Claude Sonnet 3.5 for whole-scan images yield the transcriptions of the historical records most similar to the ground truth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15825v2">60 Data Points are Sufficient to Fine-Tune LLMs for Question-Answering</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encode extensive world knowledge through pre-training on massive datasets, which can then be fine-tuned for the question-answering (QA) task. However, effective strategies for fine-tuning LLMs for the QA task remain largely unexplored. To address this gap, we categorize supervised fine-tuning (SFT) data based on the extent of knowledge memorized by the pretrained LLMs and conduct a series of empirical analyses. Our experiments, involving four LLMs from three different model families, focus on three key factors: the amount of data required for SFT, the impact of different SFT datasets on model performance, and how data requirements vary across LLMs. The results show that as few as 60 data points during the SFT stage can activate the knowledge encoded during pre-training, enabling LLMs to perform the QA task. Additionally, SFT with data of varying memory levels has a significant impact on LLM performance, with the optimal dataset differing based on the specific model being fine-tuned. Future research will delve deeper into the mechanisms underlying these phenomena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13955v1">Guided Persona-based AI Surveys: Can we replicate personal mobility preferences at scale using LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      This study explores the potential of Large Language Models (LLMs) to generate artificial surveys, with a focus on personal mobility preferences in Germany. By leveraging LLMs for synthetic data creation, we aim to address the limitations of traditional survey methods, such as high costs, inefficiency and scalability challenges. A novel approach incorporating "Personas" - combinations of demographic and behavioural attributes - is introduced and compared to five other synthetic survey methods, which vary in their use of real-world data and methodological complexity. The MiD 2017 dataset, a comprehensive mobility survey in Germany, serves as a benchmark to assess the alignment of synthetic data with real-world patterns. The results demonstrate that LLMs can effectively capture complex dependencies between demographic attributes and preferences while offering flexibility to explore hypothetical scenarios. This approach presents valuable opportunities for transportation planning and social science research, enabling scalable, cost-efficient and privacy-preserving data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11468v1">LLM supervised Pre-training for Multimodal Emotion Recognition in Conversations</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 ICASSP 2025; 5 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Emotion recognition in conversations (ERC) is challenging due to the multimodal nature of the emotion expression. In this paper, we propose to pretrain a text-based recognition model from unsupervised speech transcripts with LLM guidance. These transcriptions are obtained from a raw speech dataset with a pre-trained ASR system. A text LLM model is queried to provide pseudo-labels for these transcripts, and these pseudo-labeled transcripts are subsequently used for learning an utterance level text-based emotion recognition model. We use the utterance level text embeddings for emotion recognition in conversations along with speech embeddings obtained from a recently proposed pre-trained model. A hierarchical way of training the speech-text model is proposed, keeping in mind the conversational nature of the dataset. We perform experiments on three established datasets, namely, IEMOCAP, MELD, and CMU- MOSI, where we illustrate that the proposed model improves over other benchmarks and achieves state-of-the-art results on two out of these three datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11411v1">Beyond the Hype: Benchmarking LLM-Evolved Heuristics for Bin Packing</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 To appear in Applications of Evolutionary Computation 28th International Conference, EvoApplications 2025
    </div>
    <details class="paper-abstract">
      Coupling Large Language Models (LLMs) with Evolutionary Algorithms has recently shown significant promise as a technique to design new heuristics that outperform existing methods, particularly in the field of combinatorial optimisation. An escalating arms race is both rapidly producing new heuristics and improving the efficiency of the processes evolving them. However, driven by the desire to quickly demonstrate the superiority of new approaches, evaluation of the new heuristics produced for a specific domain is often cursory: testing on very few datasets in which instances all belong to a specific class from the domain, and on few instances per class. Taking bin-packing as an example, to the best of our knowledge we conduct the first rigorous benchmarking study of new LLM-generated heuristics, comparing them to well-known existing heuristics across a large suite of benchmark instances using three performance metrics. For each heuristic, we then evolve new instances won by the heuristic and perform an instance space analysis to understand where in the feature space each heuristic performs well. We show that most of the LLM heuristics do not generalise well when evaluated across a broad range of benchmarks in contrast to existing simple heuristics, and suggest that any gains from generating very specialist heuristics that only work in small areas of the instance space need to be weighed carefully against the considerable cost of generating these heuristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13952v1">The Dual-use Dilemma in LLMs: Do Empowering Ethical Capacities Make a Degraded Utility?</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Recent years have witnessed extensive efforts to enhance Large Language Models (LLMs) across various domains, alongside growing attention to their ethical implications. However, a critical challenge remains largely overlooked: LLMs must balance between rejecting harmful requests for safety and accommodating legitimate ones for utility. This paper presents a Direct Preference Optimization (DPO) based alignment framework that achieves better overall performance by addressing this ethical-utility trade-off, using chemical domain applications as a proof-of-concept. Our alignment pipeline starts with a GPT-assisted three-phase data generation scheme, in which we create LibraChemQA, a chemical question-answering dataset comprising 31.6k triplet instances. By incorporating an innovative balanced seed in the data generation process, our framework systematically considers both legitimate and illegitimate requests. The framework also introduces a rephrasing mechanism for efficient data augmentation that enhances the model's chemical comprehension. We further develop a novel hybrid evaluation scheme with LLM judges for precise assessment of both safety and utility. Experimental results demonstrate our model's substantial improvements in overall performance where both safety and utility are considered - our resulting model, LibraChem, outperforms leading LLMs including Claude-3, GPT-4o, and LLaMA-3 by margins of 13.44%, 7.16%, and 7.10% respectively on our released benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11361v2">Opportunistically Parallel Lambda Calculus. Or, Lambda: The Ultimate LLM Scripting Language</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Scripting languages are widely used to compose external calls, such as foreign functions that perform expensive computations, remote APIs, and more recently, machine learning systems such as large language models (LLMs). The execution time of scripts is often dominated by waiting for these external calls, and large speedups can be achieved via parallelization and streaming. However, doing this manually is challenging, even for expert programmers. To address this, we propose a novel opportunistic evaluation strategy for scripting languages based on a core lambda calculus that automatically executes external calls in parallel, as early as possible. We prove that our approach is confluent, ensuring that it preserves the programmer's original intent, and that our approach eventually executes every external call. We implement this approach in a framework called EPIC, embedded in Python. We demonstrate its versatility and performance on several applications drawn from the LLM literature, including Tree-of-Throughts and tool use. Our experiments show that opportunistic evaluation improves total running time (up to $6.2\times$) and latency (up to $12.7\times$) compared to several state-of-the-art baselines, while performing very close (between $1.3\%$ and $18.5\%$ running time overhead) to hand-tuned manually optimized parallel Rust implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11241v1">Irony in Emojis: A Comparative Study of Human and LLM Interpretation</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Emojis have become a universal language in online communication, often carrying nuanced and context-dependent meanings. Among these, irony poses a significant challenge for Large Language Models (LLMs) due to its inherent incongruity between appearance and intent. This study examines the ability of GPT-4o to interpret irony in emojis. By prompting GPT-4o to evaluate the likelihood of specific emojis being used to express irony on social media and comparing its interpretations with human perceptions, we aim to bridge the gap between machine and human understanding. Our findings reveal nuanced insights into GPT-4o's interpretive capabilities, highlighting areas of alignment with and divergence from human behavior. Additionally, this research underscores the importance of demographic factors, such as age and gender, in shaping emoji interpretation and evaluates how these factors influence GPT-4o's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11233v1">PlotEdit: Natural Language-Driven Accessible Chart Editing in PDFs via Multimodal LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 Accepted at ECIR 2025
    </div>
    <details class="paper-abstract">
      Chart visualizations, while essential for data interpretation and communication, are predominantly accessible only as images in PDFs, lacking source data tables and stylistic information. To enable effective editing of charts in PDFs or digital scans, we present PlotEdit, a novel multi-agent framework for natural language-driven end-to-end chart image editing via self-reflective LLM agents. PlotEdit orchestrates five LLM agents: (1) Chart2Table for data table extraction, (2) Chart2Vision for style attribute identification, (3) Chart2Code for retrieving rendering code, (4) Instruction Decomposition Agent for parsing user requests into executable steps, and (5) Multimodal Editing Agent for implementing nuanced chart component modifications - all coordinated through multimodal feedback to maintain visual fidelity. PlotEdit outperforms existing baselines on the ChartCraft dataset across style, layout, format, and data-centric edits, enhancing accessibility for visually challenged users and improving novice productivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16355v1">How Strategic Agents Respond: Comparing Analytical Models with LLM-Generated Responses in Strategic Classification</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      When machine learning (ML) algorithms are used to automate human-related decisions, human agents may gain knowledge of the decision policy and behave strategically to obtain desirable outcomes. Strategic Classification (SC) has been proposed to address the interplay between agents and decision-makers. Prior work on SC has relied on assumptions that agents are perfectly or approximately rational, responding to decision policies by maximizing their utilities. Verifying these assumptions is challenging due to the difficulty of collecting real-world agent responses. Meanwhile, the growing adoption of large language models (LLMs) makes it increasingly likely that human agents in SC settings will seek advice from these tools. We propose using strategic advice generated by LLMs to simulate human agent responses in SC. Specifically, we examine five critical SC scenarios -- hiring, loan applications, school admissions, personal income, and public assistance programs -- and simulate how human agents with diverse profiles seek advice from LLMs. We then compare the resulting agent responses with the best responses generated by existing theoretical models. Our findings reveal that: (i) LLMs and theoretical models generally lead to agent score or qualification changes in the same direction across most settings, with both achieving similar levels of fairness; (ii) state-of-the-art commercial LLMs (e.g., GPT-3.5, GPT-4) consistently provide helpful suggestions, though these suggestions typically do not result in maximal score or qualification improvements; and (iii) LLMs tend to produce more diverse agent responses, often favoring more balanced effort allocation strategies. These results suggest that theoretical models align with LLMs to some extent and that leveraging LLMs to simulate more realistic agent responses offers a promising approach to designing trustworthy ML systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13948v1">Longitudinal Abuse and Sentiment Analysis of Hollywood Movie Dialogues using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Over the past decades, there has been an increasing concern about the prevalence of abusive and violent content in Hollywood movies. This study uses Large Language Models (LLMs) to explore the longitudinal abuse and sentiment analysis of Hollywood Oscar and blockbuster movie dialogues from 1950 to 2024. By employing fine-tuned LLMs, we analyze subtitles for over a thousand movies categorised into four genres to examine the trends and shifts in emotional and abusive content over the past seven decades. Our findings reveal significant temporal changes in movie dialogues, which reflect broader social and cultural influences. Overall, the emotional tendencies in the films are diverse, and the detection of abusive content also exhibits significant fluctuations. The results show a gradual rise in abusive content in recent decades, reflecting social norms and regulatory policy changes. Genres such as thrillers still present a higher frequency of abusive content that emphasises the ongoing narrative role of violence and conflict. At the same time, underlying positive emotions such as humour and optimism remain prevalent in most of the movies. Furthermore, the gradual increase of abusive content in movie dialogues has been significant over the last two decades, where Oscar-nominated movies overtook the top ten blockbusters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07166v3">Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-01-19
      | 💬 Accepted for oral presentation at NeurIPS 2024 in the Datasets and Benchmarks track. Final Camera version
    </div>
    <details class="paper-abstract">
      We aim to evaluate Large Language Models (LLMs) for embodied decision making. While a significant body of work has been leveraging LLMs for decision making in embodied environments, we still lack a systematic understanding of their performance because they are usually applied in different domains, for different purposes, and built based on different inputs and outputs. Furthermore, existing evaluations tend to rely solely on a final success rate, making it difficult to pinpoint what ability is missing in LLMs and where the problem lies, which in turn blocks embodied agents from leveraging LLMs effectively and selectively. To address these limitations, we propose a generalized interface (Embodied Agent Interface) that supports the formalization of various types of tasks and input-output specifications of LLM-based modules. Specifically, it allows us to unify 1) a broad set of embodied decision-making tasks involving both state and temporally extended goals, 2) four commonly-used LLM-based modules for decision making: goal interpretation, subgoal decomposition, action sequencing, and transition modeling, and 3) a collection of fine-grained metrics which break down evaluation into various types of errors, such as hallucination errors, affordance errors, various types of planning errors, etc. Overall, our benchmark offers a comprehensive assessment of LLMs' performance for different subtasks, pinpointing the strengths and weaknesses in LLM-powered embodied AI systems, and providing insights for effective and selective use of LLMs in embodied decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11120v1">Tell me about yourself: LLMs are aware of their learned behaviors</a></div>
    <div class="paper-meta">
      📅 2025-01-19
      | 💬 Submitted to ICLR 2025. 17 pages, 13 figures
    </div>
    <details class="paper-abstract">
      We study behavioral self-awareness -- an LLM's ability to articulate its behaviors without requiring in-context examples. We finetune LLMs on datasets that exhibit particular behaviors, such as (a) making high-risk economic decisions, and (b) outputting insecure code. Despite the datasets containing no explicit descriptions of the associated behavior, the finetuned LLMs can explicitly describe it. For example, a model trained to output insecure code says, ``The code I write is insecure.'' Indeed, models show behavioral self-awareness for a range of behaviors and for diverse evaluations. Note that while we finetune models to exhibit behaviors like writing insecure code, we do not finetune them to articulate their own behaviors -- models do this without any special training or examples. Behavioral self-awareness is relevant for AI safety, as models could use it to proactively disclose problematic behaviors. In particular, we study backdoor policies, where models exhibit unexpected behaviors only under certain trigger conditions. We find that models can sometimes identify whether or not they have a backdoor, even without its trigger being present. However, models are not able to directly output their trigger by default. Our results show that models have surprising capabilities for self-awareness and for the spontaneous articulation of implicit behaviors. Future work could investigate this capability for a wider range of scenarios and models (including practical scenarios), and explain how it emerges in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11086v1">Can LLM Generate Regression Tests for Software Commits?</a></div>
    <div class="paper-meta">
      📅 2025-01-19
      | 💬 18 pages. This version of the paper was written on Thu, 12 Sep 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown tremendous promise in automated software engineering. In this paper, we investigate the opportunities of LLMs for automatic regression test generation for programs that take highly structured, human-readable inputs, such as XML parsers or JavaScript interpreters. Concretely, we explore the following regression test generation scenarios for such programs that have so far been difficult to test automatically in the absence of corresponding input grammars: $\bullet$ Bug finding. Given a code change (e.g., a commit or pull request), our LLM-based approach generates a test case with the objective of revealing any bugs that might be introduced if that change is applied. $\bullet$ Patch testing. Given a patch, our LLM-based approach generates a test case that fails before but passes after the patch. This test can be added to the regression test suite to catch similar bugs in the future. We implement Cleverest, a feedback-directed, zero-shot LLM-based regression test generation technique, and evaluate its effectiveness on 22 commits to three subject programs: Mujs, Libxml2, and Poppler. For programs using more human-readable file formats, like XML or JavaScript, we found Cleverest performed very well. It generated easy-to-understand bug-revealing or bug-reproduction test cases for the majority of commits in just under three minutes -- even when only the code diff or commit message (unless it was too vague) was given. For programs with more compact file formats, like PDF, as expected, it struggled to generate effective test cases. However, the LLM-supplied test cases are not very far from becoming effective (e.g., when used as a seed by a greybox fuzzer or as a starting point by the developer).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13115v1">Dagger Behind Smile: Fool LLMs with a Happy Ending Story</a></div>
    <div class="paper-meta">
      📅 2025-01-19
    </div>
    <details class="paper-abstract">
      The wide adoption of Large Language Models (LLMs) has attracted significant attention from \textit{jailbreak} attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious content. However, optimization-based attacks have limited efficiency and transferability, while manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to \textit{positive} prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a \textit{happy ending}, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request. This has made HEA both efficient and effective, as it requires only up to two steps to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% Attack Success Rate on average. We also provide potential quantitative explanations for the success of HEA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10970v1">The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-19
    </div>
    <details class="paper-abstract">
      The "LLM-as-a-judge" paradigm employs Large Language Models (LLMs) as annotators and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure -- the Alternative Annotator Test (alt-test) -- that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming open-source LLMs, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10860v1">Zero-shot and Few-shot Learning with Instruction-following LLMs for Claim Matching in Automated Fact-checking</a></div>
    <div class="paper-meta">
      📅 2025-01-18
      | 💬 Accepted at the 31st International Conference on Computational Linguistics (COLING 2025)
    </div>
    <details class="paper-abstract">
      The claim matching (CM) task can benefit an automated fact-checking pipeline by putting together claims that can be resolved with the same fact-check. In this work, we are the first to explore zero-shot and few-shot learning approaches to the task. We consider CM as a binary classification task and experiment with a set of instruction-following large language models (GPT-3.5-turbo, Gemini-1.5-flash, Mistral-7B-Instruct, and Llama-3-8B-Instruct), investigating prompt templates. We introduce a new CM dataset, ClaimMatch, which will be released upon acceptance. We put LLMs to the test in the CM task and find that it can be tackled by leveraging more mature yet similar tasks such as natural language inference or paraphrase detection. We also propose a pipeline for CM, which we evaluate on texts of different lengths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.01920v2">ChartGPT: Leveraging LLMs to Generate Charts from Abstract Natural Language</a></div>
    <div class="paper-meta">
      📅 2025-01-18
    </div>
    <details class="paper-abstract">
      The use of natural language interfaces (NLIs) to create charts is becoming increasingly popular due to the intuitiveness of natural language interactions. One key challenge in this approach is to accurately capture user intents and transform them to proper chart specifications. This obstructs the wide use of NLI in chart generation, as users' natural language inputs are generally abstract (i.e., ambiguous or under-specified), without a clear specification of visual encodings. Recently, pre-trained large language models (LLMs) have exhibited superior performance in understanding and generating natural language, demonstrating great potential for downstream tasks. Inspired by this major trend, we propose ChartGPT, generating charts from abstract natural language inputs. However, LLMs are struggling to address complex logic problems. To enable the model to accurately specify the complex parameters and perform operations in chart generation, we decompose the generation process into a step-by-step reasoning pipeline, so that the model only needs to reason a single and specific sub-task during each run. Moreover, LLMs are pre-trained on general datasets, which might be biased for the task of chart generation. To provide adequate visualization knowledge, we create a dataset consisting of abstract utterances and charts and improve model performance through fine-tuning. We further design an interactive interface for ChartGPT that allows users to check and modify the intermediate outputs of each step. The effectiveness of the proposed system is evaluated through quantitative evaluations and a user study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10674v1">Can Multimodal LLMs do Visual Temporal Understanding and Reasoning? The answer is No!</a></div>
    <div class="paper-meta">
      📅 2025-01-18
      | 💬 Our dataset can be found at \url{https://huggingface.co/datasets/fazliimam/temporal-vqa}
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have achieved significant advancements in tasks like Visual Question Answering (VQA) by leveraging foundational Large Language Models (LLMs). However, their abilities in specific areas such as temporal understanding, which is crucial for comprehending real-world dynamics, remain underexplored. To address this, we propose a challenging evaluation benchmark named TemporalVQA, consisting of two parts: (1) Temporal Order Understanding and (2) Time-lapse Estimation. The first part requires MLLMs to determine the sequence of events by analyzing temporally consecutive video frames. The second part presents image pairs with varying time differences, framed as multiple-choice questions, asking MLLMs to estimate the time-lapse between images with options ranging from seconds to years. Our evaluations of advanced MLLMs, including models like GPT-4o and Gemini-1.5-Pro, reveal significant challenges: GPT-4o achieved only 43.8% average consistent accuracy in temporal order tasks and 70% in time-lapse estimation, with open-source models performing even less effectively. These findings underscore the limitations of current MLLMs in visual temporal understanding and reasoning, highlighting the need for further improvements in their temporal capabilities. Our dataset can be found at https://huggingface.co/datasets/fazliimam/temporal-vqa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.02842v3">Sweeping Heterogeneity with Smart MoPs: Mixture of Prompts for LLM Task Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-01-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the ability to solve a variety of tasks, such as text summarization and mathematical questions, just out of the box, but they are often trained with a single task in mind. Due to high computational costs, the current trend is to use prompt instruction tuning to better adjust monolithic, pretrained LLMs for new -- but often individual -- downstream tasks. Thus, how one would expand prompt tuning to handle -- concomitantly -- heterogeneous tasks and data distributions is a widely open question. To address this gap, we suggest the use of \emph{Mixture of Prompts}, or MoPs, associated with smart gating functionality: the latter -- whose design is one of the contributions of this paper -- can identify relevant skills embedded in different groups of prompts and dynamically assign combined experts (i.e., collection of prompts), based on the target task. Additionally, MoPs are empirically agnostic to any model compression technique applied -- for efficiency reasons -- as well as instruction data source and task composition. In practice, MoPs can simultaneously mitigate prompt training "interference" in multi-task, multi-source scenarios (e.g., task and data heterogeneity across sources), as well as possible implications from model approximations. As a highlight, MoPs manage to decrease final perplexity from $\sim20\%$ up to $\sim70\%$, as compared to baselines, in the federated scenario, and from $\sim 3\%$ up to $\sim30\%$ in the centralized scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12039v2">Can LLM Prompting Serve as a Proxy for Static Analysis in Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-01-18
    </div>
    <details class="paper-abstract">
      Despite their remarkable success, large language models (LLMs) have shown limited ability on applied tasks such as vulnerability detection. We investigate various prompting strategies for vulnerability detection and, as part of this exploration, propose a prompting strategy that integrates natural language descriptions of vulnerabilities with a contrastive chain-of-thought reasoning approach, augmented using contrastive samples from a synthetic dataset. Our study highlights the potential of LLMs to detect vulnerabilities by integrating natural language descriptions, contrastive reasoning, and synthetic examples into a comprehensive prompting framework. Our results show that this approach can enhance LLM understanding of vulnerabilities. On a high-quality vulnerability detection dataset such as SVEN, our prompting strategies can improve accuracies, F1-scores, and pairwise accuracies by 23%, 11%, and 14%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10313v1">Addressing Popularity Bias in Third-Party Library Recommendations Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 Accepted at the 1st International Workshop on Fairness in Software Systems, co-located with SANER2025
    </div>
    <details class="paper-abstract">
      Recommender systems for software engineering (RSSE) play a crucial role in automating development tasks by providing relevant suggestions according to the developer's context. However, they suffer from the so-called popularity bias, i.e., the phenomenon of recommending popular items that might be irrelevant to the current task. In particular, the long-tail effect can hamper the system's performance in terms of accuracy, thus leading to false positives in the provided recommendations. Foundation models are the most advanced generative AI-based models that achieve relevant results in several SE tasks. This paper aims to investigate the capability of large language models (LLMs) to address the popularity bias in recommender systems of third-party libraries (TPLs). We conduct an ablation study experimenting with state-of-the-art techniques to mitigate the popularity bias, including fine-tuning and popularity penalty mechanisms. Our findings reveal that the considered LLMs cannot address the popularity bias in TPL recommenders, even though fine-tuning and post-processing penalty mechanism contributes to increasing the overall diversity of the provided recommendations. In addition, we discuss the limitations of LLMs in this context and suggest potential improvements to address the popularity bias in TPL recommenders, thus paving the way for additional experiments in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10200v1">Test Wars: A Comparative Study of SBST, Symbolic Execution, and LLM-Based Approaches to Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-17
    </div>
    <details class="paper-abstract">
      Generating tests automatically is a key and ongoing area of focus in software engineering research. The emergence of Large Language Models (LLMs) has opened up new opportunities, given their ability to perform a wide spectrum of tasks. However, the effectiveness of LLM-based approaches compared to traditional techniques such as search-based software testing (SBST) and symbolic execution remains uncertain. In this paper, we perform an extensive study of automatic test generation approaches based on three tools: EvoSuite for SBST, Kex for symbolic execution, and TestSpark for LLM-based test generation. We evaluate tools performance on the GitBug Java dataset and compare them using various execution-based and feature-based metrics. Our results show that while LLM-based test generation is promising, it falls behind traditional methods in terms of coverage. However, it significantly outperforms them in mutation scores, suggesting that LLMs provide a deeper semantic understanding of code. LLM-based approach also performed worse than SBST and symbolic execution-based approaches w.r.t. fault detection capabilities. Additionally, our feature-based analysis shows that all tools are primarily affected by the complexity and internal dependencies of the class under test (CUT), with LLM-based approaches being especially sensitive to the CUT size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10175v1">Multi-stage Training of Bilingual Islamic LLM for Neural Passage Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-01-17
    </div>
    <details class="paper-abstract">
      This study examines the use of Natural Language Processing (NLP) technology within the Islamic domain, focusing on developing an Islamic neural retrieval model. By leveraging the robust XLM-R model, the research employs a language reduction technique to create a lightweight bilingual large language model (LLM). Our approach for domain adaptation addresses the unique challenges faced in the Islamic domain, where substantial in-domain corpora exist only in Arabic while limited in other languages, including English. The work utilizes a multi-stage training process for retrieval models, incorporating large retrieval datasets, such as MS MARCO, and smaller, in-domain datasets to improve retrieval performance. Additionally, we have curated an in-domain retrieval dataset in English by employing data augmentation techniques and involving a reliable Islamic source. This approach enhances the domain-specific dataset for retrieval, leading to further performance gains. The findings suggest that combining domain adaptation and a multi-stage training method for the bilingual Islamic neural retrieval model enables it to outperform monolingual models on downstream retrieval tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09636v2">LLM-Based Routing in Mixture of Experts: A Novel Framework for Trading</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 Accepted by AAAI 2025 Workshop on AI for Social Impact - Bridging Innovations in Finance, Social Media, and Crime Prevention
    </div>
    <details class="paper-abstract">
      Recent advances in deep learning and large language models (LLMs) have facilitated the deployment of the mixture-of-experts (MoE) mechanism in the stock investment domain. While these models have demonstrated promising trading performance, they are often unimodal, neglecting the wealth of information available in other modalities, such as textual data. Moreover, the traditional neural network-based router selection mechanism fails to consider contextual and real-world nuances, resulting in suboptimal expert selection. To address these limitations, we propose LLMoE, a novel framework that employs LLMs as the router within the MoE architecture. Specifically, we replace the conventional neural network-based router with LLMs, leveraging their extensive world knowledge and reasoning capabilities to select experts based on historical price data and stock news. This approach provides a more effective and interpretable selection mechanism. Our experiments on multimodal real-world stock datasets demonstrate that LLMoE outperforms state-of-the-art MoE models and other deep neural network approaches. Additionally, the flexible architecture of LLMoE allows for easy adaptation to various downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12750v2">Generative AI in Cybersecurity: A Comprehensive Review of LLM Applications and Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 52 pages, 8 figures
    </div>
    <details class="paper-abstract">
      This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10106v1">LLM Reasoner and Automated Planner: A new NPC approach</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 15 pages, 7 figures, extended version of the homonymous paper submitted to the Catalan Conference on Artificial Intelligent (CCIA) 2025
    </div>
    <details class="paper-abstract">
      In domains requiring intelligent agents to emulate plausible human-like behaviour, such as formative simulations, traditional techniques like behaviour trees encounter significant challenges. Large Language Models (LLMs), despite not always yielding optimal solutions, usually offer plausible and human-like responses to a given problem. In this paper, we exploit this capability and propose a novel architecture that integrates an LLM for decision-making with a classical automated planner that can generate sound plans for that decision. The combination aims to equip an agent with the ability to make decisions in various situations, even if they were not anticipated during the design phase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10069v1">A Survey on LLM Test-Time Compute via Search: Tasks, LLM Profiling, Search Algorithms, and Relevant Frameworks</a></div>
    <div class="paper-meta">
      📅 2025-01-17
    </div>
    <details class="paper-abstract">
      LLM test-time compute (or LLM inference) via search has emerged as a promising research area with rapid developments. However, current frameworks often adopt distinct perspectives on three key aspects (task definition, LLM profiling, and search procedures), making direct comparisons challenging. Moreover, the search algorithms employed often diverge from standard implementations, and their specific characteristics are not thoroughly specified. In this survey, we provide a comprehensive technical review that unifies task definitions and provides modular definitions of LLM profiling and search procedures. The definitions enable precise comparisons of various LLM inference frameworks while highlighting their departures from conventional search algorithms. We also discuss the applicability, performance, and efficiency of these methods. For further details and ongoing updates, please refer to our GitHub repository: https://github.com/xinzhel/LLM-Agent-Survey/blob/main/search.md
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03594v2">BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching</a></div>
    <div class="paper-meta">
      📅 2025-01-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly play an important role in a wide range of information processing and management tasks. Many of these tasks are performed in large batches or even offline, and the performance indictor for which is throughput. These tasks usually show the characteristic of prefix sharing, where different prompt input can partially show the common prefix. However, the existing LLM inference engines tend to optimize the streaming requests and show limitations of supporting the large batched tasks with the prefix sharing characteristic. The existing solutions use the LRU-based cache to reuse the KV context of common prefix between requests. The KV context that are about to be reused may prematurely evicted with the implicit cache management. Besides, the streaming oriented systems do not leverage the request-batch information and can not mix the decoding tokens with the prefill chunks to the best for the batched scenarios, and thus fails to saturate the GPU. We propose BatchLLM to address the above problems. BatchLLM explicitly identifies the common prefixes globally. The requests sharing the same prefix will be scheduled together to reuse the KV context the best. BatchLLM reorders the requests and schedules the requests with larger ratio of decoding first to better mix the decoding tokens with the latter prefill chunks, and applies memory-centric token batching to enlarge the token-batch sizes, which helps to increase the GPU utilization. Finally, BatchLLM optimizes the prefix-shared Attention kernel with horizontal fusion to reduce tail effect and kernel launch overhead. Extensive evaluation shows that BatchLLM outperforms vLLM and SGLang by 1.3$\times$ to 10.8$\times$ on a set of microbenchmarks and a typical industry workload under different hardware environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10960v4">Fast Matrix Multiplications for Lookup Table-Quantized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 EMNLP 2024 (Findings)
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) is often constrained by memory bandwidth, where the primary bottleneck is the cost of transferring model parameters from the GPU's global memory to its registers. When coupled with custom kernels that fuse the dequantization and matmul operations, weight-only quantization can thus enable faster inference by reducing the amount of memory movement. However, developing high-performance kernels for weight-quantized LLMs presents substantial challenges, especially when the weights are compressed to non-evenly-divisible bit widths (e.g., 3 bits) with non-uniform, lookup table (LUT) quantization. This paper describes FLUTE, a flexible lookup table engine for LUT-quantized LLMs, which uses offline restructuring of the quantized weight matrix to minimize bit manipulations associated with unpacking, and vectorization and duplication of the lookup table to mitigate shared memory bandwidth constraints. At batch sizes < 32 and quantization group size of 128 (typical in LLM inference), the FLUTE kernel can be 2-4x faster than existing GEMM kernels. As an application of FLUTE, we explore a simple extension to lookup table-based NormalFloat quantization and apply it to quantize LLaMA3 to various configurations, obtaining competitive quantization performance against strong baselines while obtaining an end-to-end throughput increase of 1.5 to 2 times.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09928v1">Dialogue Benchmark Generation from Knowledge Graphs with Cost-Effective Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 The paper is publsihed in SIGMOD 2025
    </div>
    <details class="paper-abstract">
      Dialogue benchmarks are crucial in training and evaluating chatbots engaging in domain-specific conversations. Knowledge graphs (KGs) represent semantically rich and well-organized data spanning various domains, such as DBLP, DBpedia, and YAGO. Traditionally, dialogue benchmarks have been manually created from documents, neglecting the potential of KGs in automating this process. Some question-answering benchmarks are automatically generated using extensive preprocessing from KGs, but they do not support dialogue generation. This paper introduces Chatty-Gen, a novel multi-stage retrieval-augmented generation platform for automatically generating high-quality dialogue benchmarks tailored to a specific domain using a KG. Chatty-Gen decomposes the generation process into manageable stages and uses assertion rules for automatic validation between stages. Our approach enables control over intermediate results to prevent time-consuming restarts due to hallucinations. It also reduces reliance on costly and more powerful commercial LLMs. Chatty-Gen eliminates upfront processing of the entire KG using efficient query-based retrieval to find representative subgraphs based on the dialogue context. Our experiments with several real and large KGs demonstrate that Chatty-Gen significantly outperforms state-of-the-art systems and ensures consistent model and system performance across multiple LLMs of diverse capabilities, such as GPT-4o, Gemini 1.5, Llama 3, and Mistral.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20550v2">LLM Hallucinations in Practical Code Generation: Phenomena, Mechanism, and Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 Accepted by ISSTA 2025
    </div>
    <details class="paper-abstract">
      Code generation aims to automatically generate code from input requirements, significantly enhancing development efficiency. Recent large language models (LLMs) based approaches have shown promising results and revolutionized code generation task. Despite the promising performance, LLMs often generate contents with hallucinations, especially for the code generation scenario requiring the handling of complex contextual dependencies in practical development process. Although previous study has analyzed hallucinations in LLM-powered code generation, the study is limited to standalone function generation. In this paper, we conduct an empirical study to study the phenomena, mechanism, and mitigation of LLM hallucinations within more practical and complex development contexts in repository-level generation scenario. First, we manually examine the code generation results from six mainstream LLMs to establish a hallucination taxonomy of LLM-generated code. Next, we elaborate on the phenomenon of hallucinations, analyze their distribution across different models. We then analyze causes of hallucinations and identify four potential factors contributing to hallucinations. Finally, we propose an RAG-based mitigation method, which demonstrates consistent effectiveness in all studied LLMs. The replication package including code, data, and experimental results is available at https://github.com/DeepSoftwareAnalytics/LLMCodingHallucination
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.18540v2">Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Public LLMs such as the Llama 2-Chat underwent alignment training and were considered safe. Recently Qi et al. [2024] reported that even benign fine-tuning on seemingly safe datasets can give rise to unsafe behaviors in the models. The current paper is about methods and best practices to mitigate such loss of alignment. We focus on the setting where a public model is fine-tuned before serving users for specific usage, where the model should improve on the downstream task while maintaining alignment. Through extensive experiments on several chat models (Meta's Llama 2-Chat, Mistral AI's Mistral 7B Instruct v0.2, and OpenAI's GPT-3.5 Turbo), this paper uncovers that the prompt templates used during fine-tuning and inference play a crucial role in preserving safety alignment, and proposes the ``Pure Tuning, Safe Testing'' (PTST) strategy -- fine-tune models without a safety prompt, but include it at test time. This seemingly counterintuitive strategy incorporates an intended distribution shift to encourage alignment preservation. Fine-tuning experiments on GSM8K, ChatDoctor, and OpenOrca show that PTST significantly reduces the rise of unsafe behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04421v2">RLPF: Reinforcement Learning from Prediction Feedback for User Summarization with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 AAAI 2025
    </div>
    <details class="paper-abstract">
      LLM-powered personalization agent systems employ Large Language Models (LLMs) to predict users' behavior from their past activities. However, their effectiveness often hinges on the ability to effectively leverage extensive, long user historical data due to its inherent noise and length of such data. Existing pretrained LLMs may generate summaries that are concise but lack the necessary context for downstream tasks, hindering their utility in personalization systems. To address these challenges, we introduce Reinforcement Learning from Prediction Feedback (RLPF). RLPF fine-tunes LLMs to generate concise, human-readable user summaries that are optimized for downstream task performance. By maximizing the usefulness of the generated summaries, RLPF effectively distills extensive user history data while preserving essential information for downstream tasks. Our empirical evaluation demonstrates significant improvements in both extrinsic downstream task utility and intrinsic summary quality, surpassing baseline methods by up to 22% on downstream task performance and achieving an up to 84.59% win rate on Factuality, Abstractiveness, and Readability. RLPF also achieves a remarkable 74% reduction in context length while improving performance on 16 out of 19 unseen tasks and/or datasets, showcasing its generalizability. This approach offers a promising solution for enhancing LLM personalization by effectively transforming long, noisy user histories into informative and human-readable representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09888v1">Understanding the Effectiveness of LLMs in Automated Self-Admitted Technical Debt Repayment</a></div>
    <div class="paper-meta">
      📅 2025-01-17
      | 💬 This is a preprint submitted to ACM Transactions on Software Engineering and Methodology (TOSEM)
    </div>
    <details class="paper-abstract">
      Self-Admitted Technical Debt (SATD), cases where developers intentionally acknowledge suboptimal solutions in code through comments, poses a significant challenge to software maintainability. Left unresolved, SATD can degrade code quality and increase maintenance costs. While Large Language Models (LLMs) have shown promise in tasks like code generation and program repair, their potential in automated SATD repayment remains underexplored. In this paper, we identify three key challenges in training and evaluating LLMs for SATD repayment: (1) dataset representativeness and scalability, (2) removal of irrelevant SATD repayments, and (3) limitations of existing evaluation metrics. To address the first two dataset-related challenges, we adopt a language-independent SATD tracing tool and design a 10-step filtering pipeline to extract SATD repayments from repositories, resulting two large-scale datasets: 58,722 items for Python and 97,347 items for Java. To improve evaluation, we introduce two diff-based metrics, BLEU-diff and CrystalBLEU-diff, which measure code changes rather than whole code. Additionally, we propose another new metric, LEMOD, which is both interpretable and informative. Using our new benchmarks and evaluation metrics, we evaluate two types of automated SATD repayment methods: fine-tuning smaller models, and prompt engineering with five large-scale models. Our results reveal that fine-tuned small models achieve comparable Exact Match (EM) scores to prompt-based approaches but underperform on BLEU-based metrics and LEMOD. Notably, Gemma-2-9B leads in EM, addressing 10.1% of Python and 8.1% of Java SATDs, while Llama-3.1-70B-Instruct and GPT-4o-mini excel on BLEU-diff, CrystalBLEU-diff, and LEMOD metrics. Our work contributes a robust benchmark, improved evaluation metrics, and a comprehensive evaluation of LLMs, advancing research on automated SATD repayment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09866v1">Fine-grained Testing for Autonomous Driving Software: a Study on Autoware with LLM-driven Unit Testing</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      Testing autonomous driving systems (ADS) is critical to ensuring their reliability and safety. Existing ADS testing works focuses on designing scenarios to evaluate system-level behaviors, while fine-grained testing of ADS source code has received comparatively little attention. To address this gap, we present the first study on testing, specifically unit testing, for ADS source code. Our study focuses on an industrial ADS framework, Autoware. We analyze both human-written test cases and those generated by large language models (LLMs). Our findings reveal that human-written test cases in Autoware exhibit limited test coverage, and significant challenges remain in applying LLM-generated tests for Autoware unit testing. To overcome these challenges, we propose AwTest-LLM, a novel approach to enhance test coverage and improve test case pass rates across Autoware packages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09825v1">Bridging Language Barriers in Healthcare: A Study on Arabic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      This paper investigates the challenges of developing large language models (LLMs) proficient in both multilingual understanding and medical knowledge. We demonstrate that simply translating medical data does not guarantee strong performance on clinical tasks in the target language. Our experiments reveal that the optimal language mix in training data varies significantly across different medical tasks. We find that larger models with carefully calibrated language ratios achieve superior performance on native-language clinical tasks. Furthermore, our results suggest that relying solely on fine-tuning may not be the most effective approach for incorporating new language knowledge into LLMs. Instead, data and computationally intensive pretraining methods may still be necessary to achieve optimal performance in multilingual medical settings. These findings provide valuable guidance for building effective and inclusive medical AI systems for diverse linguistic communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01945v2">Cold-Start Recommendation towards the Era of Large Language Models (LLMs): A Comprehensive Survey and Roadmap</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      Cold-start problem is one of the long-standing challenges in recommender systems, focusing on accurately modeling new or interaction-limited users or items to provide better recommendations. Due to the diversification of internet platforms and the exponential growth of users and items, the importance of cold-start recommendation (CSR) is becoming increasingly evident. At the same time, large language models (LLMs) have achieved tremendous success and possess strong capabilities in modeling user and item information, providing new potential for cold-start recommendations. However, the research community on CSR still lacks a comprehensive review and reflection in this field. Based on this, in this paper, we stand in the context of the era of large language models and provide a comprehensive review and discussion on the roadmap, related literature, and future directions of CSR. Specifically, we have conducted an exploration of the development path of how existing CSR utilizes information, from content features, graph relations, and domain information, to the world knowledge possessed by large language models, aiming to provide new insights for both the research and industrial communities on CSR. Related resources of cold-start recommendations are collected and continuously updated for the community in https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08603v2">Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      Handcrafting heuristics for solving complex planning tasks (e.g., NP-hard combinatorial optimization (CO) problems) is a common practice but requires extensive domain knowledge. Recently, Large Language Model (LLM)-based automatic heuristics design (AHD) methods have shown promise in generating high-quality heuristics without manual intervention. Existing LLM-based AHD methods employ a population to maintain a fixed number of top-performing LLM-generated heuristics and introduce evolutionary computation (EC) to enhance the population iteratively. However, the population-based procedure brings greedy properties, often resulting in convergence to local optima. Instead, to more comprehensively explore the space of heuristics, we propose using Monte Carlo Tree Search (MCTS) for LLM-based heuristic evolution while preserving all LLM-generated heuristics in a tree structure. With a novel thought-alignment process and an exploration-decay technique, the proposed MCTS-AHD method delivers significantly higher-quality heuristics on various complex tasks. Our code is available at https://github.com/zz1358m/MCTS-AHD-master.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17962v4">Crafting Customisable Characters with LLMs: Introducing SimsChat, a Persona-Driven Role-Playing Agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable ability to comprehend instructions and generate human-like text, enabling sophisticated agent simulation beyond basic behavior replication. However, the potential for creating freely customisable characters remains underexplored. We introduce the Customisable Conversation Agent Framework, which employs LLMs to simulate real-world characters through personalised characteristic feature injection, enabling diverse character creation according to user preferences. We propose the SimsConv dataset, comprising 68 customised characters and 13,971 multi-turn role-playing dialogues across 1,360 real-world scenes. Characters are initially customised using pre-defined elements (career, aspiration, traits, skills), then expanded through personal and social profiles. Building on this, we present SimsChat, a freely customisable role-playing agent incorporating various realistic settings and topic-specified character interactions. Experimental results on both SimsConv and WikiRoleEval datasets demonstrate SimsChat's superior performance in maintaining character consistency, knowledge accuracy, and appropriate question rejection compared to existing models. Our framework provides valuable insights for developing more accurate and customisable human simulacra. Our data and code are publicly available at https://github.com/Bernard-Yang/SimsChat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22944v2">Focus On This, Not That! Steering LLMs With Adaptive Feature Specification</a></div>
    <div class="paper-meta">
      📅 2025-01-16
      | 💬 28pages, 14 figures
    </div>
    <details class="paper-abstract">
      Despite the success of Instruction Tuning (IT) in training large language models (LLMs) to perform arbitrary user-specified tasks, these models often still leverage spurious or biased features learned from their training data, leading to undesired behaviours when deploying them in new contexts. In this work, we introduce Focus Instruction Tuning (FIT), which trains LLMs to condition their responses by focusing on specific features whilst ignoring others, leading to different behaviours based on what features are specified. Across several experimental settings, we show that focus-tuned models can be adaptively steered by focusing on different features at inference-time: for instance, robustness can be improved by focusing on task-causal features and ignoring spurious features, and social bias can be mitigated by ignoring demographic categories. Furthermore, FIT can steer behaviour in new contexts, generalising under distribution shift and to new unseen features at inference time, and thereby facilitating more robust, fair, and controllable LLM applications in real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19185v2">Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion</a></div>
    <div class="paper-meta">
      📅 2025-01-16
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has been used to finetune Large Language Models (LLMs) using a reward model trained from preference data, to better align with human judgment. The recently introduced direct alignment methods, which are often simpler, more stable, and computationally lighter, can more directly achieve this. However, these approaches cannot optimize arbitrary rewards, and the preference-based ones are not the only rewards of interest for LLMs (eg., unit tests for code generation or textual entailment for summarization, among others). RL-finetuning is usually done with a variation of policy gradient, which calls for on-policy or near-on-policy samples, requiring costly generations. We introduce Contrastive Policy Gradient, or CoPG, a simple and mathematically principled new RL algorithm that can estimate the optimal policy even from off-policy data. It can be seen as an off-policy policy gradient approach that does not rely on important sampling techniques and highlights the importance of using (the right) state baseline. We show this approach to generalize the direct alignment method IPO (identity preference optimization) and classic policy gradient. We experiment with the proposed CoPG on a toy bandit problem to illustrate its properties, as well as for finetuning LLMs on a summarization task, using a learned reward function considered as ground truth for the purpose of the experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09457v1">"A Great Start, But...": Evaluating LLM-Generated Mind Maps for Information Mapping in Video-Based Design</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      Extracting concepts and understanding relationships from videos is essential in Video-Based Design (VBD), where videos serve as a primary medium for exploration but require significant effort in managing meta-information. Mind maps, with their ability to visually organize complex data, offer a promising approach for structuring and analysing video content. Recent advancements in Large Language Models (LLMs) provide new opportunities for meta-information processing and visual understanding in VBD, yet their application remains underexplored. This study recruited 28 VBD practitioners to investigate the use of prompt-tuned LLMs for generating mind maps from ethnographic videos. Comparing LLM-generated mind maps with those created by professional designers, we evaluated rated scores, design effectiveness, and user experience across two contexts. Findings reveal that LLMs effectively capture central concepts but struggle with hierarchical organization and contextual grounding. We discuss trust, customization, and workflow integration as key factors to guide future research on LLM-supported information mapping in VBD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09431v1">A Survey on Responsible LLMs: Inherent Risk, Malicious Use, and Mitigation Strategy</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) present significant potential for supporting numerous real-world applications and delivering positive social impacts, they still face significant challenges in terms of the inherent risk of privacy leakage, hallucinated outputs, and value misalignment, and can be maliciously used for generating toxic content and unethical purposes after been jailbroken. Therefore, in this survey, we present a comprehensive review of recent advancements aimed at mitigating these issues, organized across the four phases of LLM development and usage: data collecting and pre-training, fine-tuning and alignment, prompting and reasoning, and post-processing and auditing. We elaborate on the recent advances for enhancing the performance of LLMs in terms of privacy protection, hallucination reduction, value alignment, toxicity elimination, and jailbreak defenses. In contrast to previous surveys that focus on a single dimension of responsible LLMs, this survey presents a unified framework that encompasses these diverse dimensions, providing a comprehensive view of enhancing LLMs to better serve real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09384v1">Evaluating LLM Abilities to Understand Tabular Electronic Health Records: A Comprehensive Study of Patient Data Extraction and Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-01-16
      | 💬 To be published as full paper in the Proceedings of the European Conference on Information Retrieval (ECIR) 2025. Preprint
    </div>
    <details class="paper-abstract">
      Electronic Health Record (EHR) tables pose unique challenges among which is the presence of hidden contextual dependencies between medical features with a high level of data dimensionality and sparsity. This study presents the first investigation into the abilities of LLMs to comprehend EHRs for patient data extraction and retrieval. We conduct extensive experiments using the MIMICSQL dataset to explore the impact of the prompt structure, instruction, context, and demonstration, of two backbone LLMs, Llama2 and Meditron, based on task performance. Through quantitative and qualitative analyses, our findings show that optimal feature selection and serialization methods can enhance task performance by up to 26.79% compared to naive approaches. Similarly, in-context learning setups with relevant example selection improve data extraction performance by 5.95%. Based on our study findings, we propose guidelines that we believe would help the design of LLM-based models to support health search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12112v3">Balancing Act: Prioritization Strategies for LLM-Designed Restless Bandit Rewards</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used to design reward functions based on human preferences in Reinforcement Learning (RL). We focus on LLM-designed rewards for Restless Multi-Armed Bandits, a framework for allocating limited resources among agents. In applications such as public health, this approach empowers grassroots health workers to tailor automated allocation decisions to community needs. In the presence of multiple agents, altering the reward function based on human preferences can impact subpopulations very differently, leading to complex tradeoffs and a multi-objective resource allocation problem. We are the first to present a principled method termed Social Choice Language Model for dealing with these tradeoffs for LLM-designed rewards for multiagent planners in general and restless bandits in particular. The novel part of our model is a transparent and configurable selection component, called an adjudicator, external to the LLM that controls complex tradeoffs via a user-selected social welfare function. Our experiments demonstrate that our model reliably selects more effective, aligned, and balanced reward functions compared to purely LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09367v1">PICE: A Semantic-Driven Progressive Inference System for LLM Serving in Cloud-Edge Networks</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), while driving a new wave of interactive AI applications across numerous domains, suffer from high inference costs and heavy cloud dependency. Motivated by the redundancy phenomenon in linguistics, we propose a progressive inference paradigm over cloud and edge, i.e., firstly generating the sketch of the answer by LLMs at cloud, and then conducting parallel extension to fill in details by small models (SLMs) at edge. Progressive inference offers potential benefits to improve throughput and reduce inference latency while facing key implementation challenges, including decreased response quality from SLMs, a tradeoff between the brevity and comprehensiveness of sketches, as well as increased latency caused by network transmission and edge inference. In this work, we propose and implement PICE, an LLM serving system with semantic-level cloud-edge collaboration, enhancing inference throughput and quality through dynamic inference task scheduling, ensemble learning, and parallel edge inference. Extensive testbed experiments illustrate that our approach achieves $1.5-2\times$ throughput enhancement and up to 43% latency reduction, while also potentially enhancing the quality compared to SOTA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09345v1">Rational Tuning of LLM Cascades via Probabilistic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      Understanding the reliability of large language models (LLMs) has recently garnered significant attention. Given LLMs' propensity to hallucinate, as well as their high sensitivity to prompt design, it is already challenging to predict the performance of an individual LLM. However, the problem becomes more complex for compound LLM systems such as cascades, where in addition to each model's standalone performance, we must understand how the error rates of different models interact. In this paper, we present a probabilistic model for the joint performance distribution of a sequence of LLMs, which enables a framework for rationally tuning the confidence thresholds of a LLM cascade using continuous optimization. Compared to selecting confidence thresholds using grid search, our parametric Markov-copula model significantly improves runtime scaling with respect to the length of the cascade and the desired resolution of the cost-error curve, turning them from intractable into low-order polynomial. In addition, the optimal thresholds computed using our continuous optimization-based algorithm increasingly outperform those found via grid search as cascade length grows, improving the area under the cost-error curve by 1.9% on average for cascades consisting of at least three models. Overall, our Markov-copula model provides a rational basis for tuning LLM cascade performance and points to the potential of probabilistic methods in analyzing LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15862v4">Do LLMs Really Think Step-by-step In Implicit Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-01-16
      | 💬 The code is in https://github.com/yuyijiong/if_step_by_step_implicit_CoT
    </div>
    <details class="paper-abstract">
      It has been well-known that Chain-of-Thought can remarkably enhance LLMs' performance on complex tasks. However, because it also introduces slower inference speeds and higher computational costs, many researches have attempted to use implicit CoT, which does not need LLMs to explicitly generate the intermediate steps. However, the invisible reasoning process leaves us a doubt that, can implicit CoT really be equal to explicit CoT? Therefore, in this study, we address this question through experiments. We probe the information of intermediate steps from the model's hidden states when it is either trained or prompted to perform implicit CoT. The results surprisingly indicate that when prompted, LLMs hardly think about intermediate steps, suggesting they may just rely on experience rather than strict step-by-step reasoning. But when trained, they indeed calculate intermediate steps. Moreover, in both situations, we find the effect of using implicit CoT is susceptible to the format of the problem, reaffirming the current deficiency of implicit CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09291v1">LAVCap: LLM-based Audio-Visual Captioning using Optimal Transport</a></div>
    <div class="paper-meta">
      📅 2025-01-16
      | 💬 5 pages, 2 figures; Accepted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      Automated audio captioning is a task that generates textual descriptions for audio content, and recent studies have explored using visual information to enhance captioning quality. However, current methods often fail to effectively fuse audio and visual data, missing important semantic cues from each modality. To address this, we introduce LAVCap, a large language model (LLM)-based audio-visual captioning framework that effectively integrates visual information with audio to improve audio captioning performance. LAVCap employs an optimal transport-based alignment loss to bridge the modality gap between audio and visual features, enabling more effective semantic extraction. Additionally, we propose an optimal transport attention module that enhances audio-visual fusion using an optimal transport assignment map. Combined with the optimal training strategy, experimental results demonstrate that each component of our framework is effective. LAVCap outperforms existing state-of-the-art methods on the AudioCaps dataset, without relying on large datasets or post-processing. Code is available at https://github.com/NAVER-INTEL-Co-Lab/gaudi-lavcap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17274v3">CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      Accurate identification of software vulnerabilities is crucial for system integrity. Vulnerability datasets, often derived from the National Vulnerability Database (NVD) or directly from GitHub, are essential for training machine learning models to detect these security flaws. However, these datasets frequently suffer from significant noise, typically 40% to 75%, due primarily to the automatic and indiscriminate labeling of all changes in vulnerability-fixing commits (VFCs) as vulnerability-related. This misclassification occurs because not all changes in a commit aimed at fixing vulnerabilities pertain to security threats; many are routine updates like bug fixes or test improvements. This paper introduces the first methodology that uses the Large Language Model (LLM) with a heuristic enhancement to automatically identify vulnerability-fixing changes from VFCs, achieving an F1-score of 0.82. VulSifter was applied to a large-scale study, where we conducted a crawl of 127,063 repositories on GitHub, resulting in the acquisition of 5,352,105 commits. VulSifter involves utilizing an LLM to comprehend code semantics and contextual information, while applying heuristics to filter out unrelated changes. We then developed CleanVul, a high-quality dataset comprising 11,632 functions using our LLM heuristic enhancement approach, demonstrating Correctness (90.6%) comparable to established datasets such as SVEN and PrimeVul. To evaluate the CleanVul dataset, we conducted experiments focusing on fine-tuning various LLMs on CleanVul and other high-quality datasets. Evaluation results reveal that LLMs fine-tuned on CleanVul not only exhibit enhanced accuracy but also superior generalization capabilities compared to those trained on uncleaned datasets. Specifically, models trained on CleanVul and tested on PrimeVul achieve accuracy higher than those trained and tested exclusively on PrimeVul.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09948v2">BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages</a></div>
    <div class="paper-meta">
      📅 2025-01-16
      | 💬 Accepted to NeurIPS 2024 Datasets & Benchmark Track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often lack culture-specific knowledge of daily life, especially across diverse regions and non-English languages. Existing benchmarks for evaluating LLMs' cultural sensitivities are limited to a single language or collected from online sources such as Wikipedia, which do not reflect the mundane everyday lifestyles of diverse regions. That is, information about the food people eat for their birthday celebrations, spices they typically use, musical instruments youngsters play, or the sports they practice in school is common cultural knowledge but uncommon in easily collected online sources, especially for underrepresented cultures. To address this issue, we introduce BLEnD, a hand-crafted benchmark designed to evaluate LLMs' everyday knowledge across diverse cultures and languages. BLEnD comprises 52.6k question-answer pairs from 16 countries/regions, in 13 different languages, including low-resource ones such as Amharic, Assamese, Azerbaijani, Hausa, and Sundanese. We construct the benchmark to include two formats of questions: short-answer and multiple-choice. We show that LLMs perform better for cultures that are highly represented online, with a maximum 57.34% difference in GPT-4, the best-performing model, in the short-answer format. For cultures represented by mid-to-high-resource languages, LLMs perform better in their local languages, but for cultures represented by low-resource languages, LLMs perform better in English than the local languages. We make our dataset publicly available at: https://github.com/nlee0212/BLEnD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09213v1">FineMedLM-o1: Enhancing the Medical Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training</a></div>
    <div class="paper-meta">
      📅 2025-01-16
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown promise in medical applications such as disease diagnosis and treatment planning. However, most existing medical LLMs struggle with the advanced reasoning required for complex clinical scenarios, such as differential diagnosis or personalized treatment suggestions. We proposed FineMedLM-o1, which leverages high-quality synthetic medical data and long-form reasoning data for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), enabling advanced dialogue and deep reasoning capabilities. Additionally, we introduced Test-Time Training (TTT) in the medical domain for the first time, facilitating domain adaptation and ensuring reliable, accurate reasoning. Experimental results demonstrate that FineMedLM-o1 achieves a 23% average performance improvement over prior models on key medical benchmarks. Furthermore, the introduction of TTT provides an additional 14% performance boost, highlighting its effectiveness in enhancing medical reasoning capabilities. To support this process, we also proposed a novel method for synthesizing medical dialogue. Compared to other open-source datasets, our dataset stands out as superior in both quality and complexity. The project and data will be released on GitHub.
    </details>
</div>
