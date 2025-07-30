# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- Part 9
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05541v1">SenseCF: LLM-Prompted Counterfactuals for Intervention and Sensor Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 In review
    </div>
    <details class="paper-abstract">
      Counterfactual explanations (CFs) offer human-centric insights into machine learning predictions by highlighting minimal changes required to alter an outcome. Therefore, CFs can be used as (i) interventions for abnormality prevention and (ii) augmented data for training robust models. In this work, we explore large language models (LLMs), specifically GPT-4o-mini, for generating CFs in a zero-shot and three-shot setting. We evaluate our approach on two datasets: the AI-Readi flagship dataset for stress prediction and a public dataset for heart disease detection. Compared to traditional methods such as DiCE, CFNOW, and NICE, our few-shot LLM-based approach achieves high plausibility (up to 99%), strong validity (up to 0.99), and competitive sparsity. Moreover, using LLM-generated CFs as augmented samples improves downstream classifier performance (an average accuracy gain of 5%), especially in low-data regimes. This demonstrates the potential of prompt-based generative techniques to enhance explainability and robustness in clinical and physiological prediction tasks. Code base: github.com/anonymous/SenseCF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18116v3">Bayesian Optimization for Controlled Image Editing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 8 figures, accept at ACL2025 Findings
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05528v1">Conversational Education at Scale: A Multi-LLM Agent Workflow for Procedural Learning and Pedagogic Quality Assessment</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced virtual educators and learners, bridging NLP with AI4Education. Existing work often lacks scalability and fails to leverage diverse, large-scale course content, with limited frameworks for assessing pedagogic quality. To this end, we propose WikiHowAgent, a multi-agent workflow leveraging LLMs to simulate interactive teaching-learning conversations. It integrates teacher and learner agents, an interaction manager, and an evaluator to facilitate procedural learning and assess pedagogic quality. We introduce a dataset of 114,296 teacher-learner conversations grounded in 14,287 tutorials across 17 domains and 727 topics. Our evaluation protocol combines computational and rubric-based metrics with human judgment alignment. Results demonstrate the workflow's effectiveness in diverse setups, offering insights into LLM capabilities across domains. Our datasets and implementations are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05504v1">Tool for Supporting Debugging and Understanding of Normative Requirements Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Normative requirements specify social, legal, ethical, empathetic, and cultural (SLEEC) norms that must be observed by a system. To support the identification of SLEEC requirements, numerous standards and regulations have been developed. These requirements are typically defined by stakeholders in the non-technical system with diverse expertise (e.g., ethicists, lawyers, social scientists). Hence, ensuring their consistency and managing the requirement elicitation process are complex and error-prone tasks. Recent research has addressed this challenge using domain-specific languages to specify normative requirements as rules, whose consistency can then be analyzed with formal methods. Nevertheless, these approaches often present the results from formal verification tools in a way that is inaccessible to non-technical users. This hinders understanding and makes the iterative process of eliciting and validating these requirements inefficient in terms of both time and effort. To address this problem, we introduce SLEEC-LLM, a tool that uses large language models (LLMs) to provide natural-language interpretations for model-checking counterexamples corresponding to SLEEC rule inconsistencies. SLEEC-LLM improves the efficiency and explainability of normative requirements elicitation and consistency analysis. To demonstrate its effectiveness, we summarise its use in two real-world case studies involving non-technical stakeholders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05461v1">GLOSS: Group of LLMs for Open-Ended Sensemaking of Passive Sensing Data for Health and Wellbeing</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      The ubiquitous presence of smartphones and wearables has enabled researchers to build prediction and detection models for various health and behavior outcomes using passive sensing data from these devices. Achieving a high-level, holistic understanding of an individual's behavior and context, however, remains a significant challenge. Due to the nature of passive sensing data, sensemaking -- the process of interpreting and extracting insights -- requires both domain knowledge and technical expertise, creating barriers for different stakeholders. Existing systems designed to support sensemaking are either not open-ended or cannot perform complex data triangulation. In this paper, we present a novel sensemaking system, Group of LLMs for Open-ended Sensemaking (GLOSS), capable of open-ended sensemaking and performing complex multimodal triangulation to derive insights. We demonstrate that GLOSS significantly outperforms the commonly used Retrieval-Augmented Generation (RAG) technique, achieving 87.93% accuracy and 66.19% consistency, compared to RAG's 29.31% accuracy and 52.85% consistency. Furthermore, we showcase the promise of GLOSS through four use cases inspired by prior and ongoing work in the UbiComp and HCI communities. Finally, we discuss the potential of GLOSS, its broader implications, and the limitations of our work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04798v3">From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Hierarchical control for robotics has long been plagued by the need to have a well defined interface layer to communicate between high-level task planners and low-level policies. With the advent of LLMs, language has been emerging as a prospective interface layer. However, this has several limitations. Not all tasks can be decomposed into steps that are easily expressible in natural language (e.g. performing a dance routine). Further, it makes end-to-end finetuning on embodied data challenging due to domain shift and catastrophic forgetting. We introduce our method -- Learnable Latent Codes as Bridges (LCB) -- as an alternate architecture to overcome these limitations. \method~uses a learnable latent code to act as a bridge between LLMs and low-level policies. This enables LLMs to flexibly communicate goals in the task plan without being entirely constrained by language limitations. Additionally, it enables end-to-end finetuning without destroying the embedding space of word tokens learned during pre-training. Through experiments on Language Table and Calvin, two common language based benchmarks for embodied agents, we find that \method~outperforms baselines (including those w/ GPT-4V) that leverage pure language as the interface layer on tasks that require reasoning and multi-step behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17172v2">What Would You Ask When You First Saw $a^2+b^2=c^2$? Evaluating LLM on Curiosity-Driven Questioning</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can store a massive amount of knowledge, yet their potential to acquire new knowledge remains unknown. We propose a novel evaluation framework that evaluates this capability. This framework prompts LLMs to generate questions about a statement introducing scientific knowledge, simulating a curious person when facing the statement for the first time. We score the qualities of the generated questions, thereby evaluating the knowledge acquisition potential of the LLM. We apply controlled ablation studies to validate our scoring procedures. Additionally, we created a synthetic dataset consisting of 1101 statements in physics, chemistry, and maths with distinct levels of difficulties, 300 general knowledge statements, and 567 incorrect statements. Human evaluations were conducted to validate our model assessments, achieving an approximate weighted Cohen's kappa of 0.7 on all three metrics considered. We find that while large models like GPT-4 and Mistral 8x7b are adept at generating coherent and relevant questions, the smaller Phi-2 model is equally or more effective. This indicates that size does not solely determine a model's knowledge acquisition potential. The proposed framework quantifies a critical model capability that was commonly overlooked and opens up research opportunities for developing more knowledgeable AI systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05403v1">PBE Meets LLM: When Few Examples Aren't Few-Shot Enough</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 7 pages, 5 figures, accepted by VLDB QDB'25 workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate code from natural language descriptions. Their performance is typically evaluated using programming benchmarks that simulate real-world tasks. These benchmarks provide specifications in the form of docstrings, function signatures, or bug reports. The model then generates a program, which is tested against predefined test cases. In contrast, Programming by Example (PBE) uses input-output examples as the specification. Traditional PBE systems rely on search-based methods over restricted transformation spaces. They are usually designed for narrow domains and fixed input formats. It remains unclear how well LLMs perform on PBE tasks. In this work, we evaluate LLMs on PBE tasks involving tabular data transformations. We prompt models to generate functions that convert an input table to an output table. We test the generated functions on unseen inputs to measure accuracy. Our study includes multiple LLMs and evaluates different prompting strategies, such as one-shot vs. multi-try. We also compare performance with and without PBE-specific knowledge. Finally, we propose a hybrid method that calls a traditional PBE solver first, and then falls back to LLMs if necessary. Our results show that LLMs support more diverse input formats and achieve higher accuracy than conventional methods. However, they struggle with tasks that contain ambiguity. The hybrid approach improves overall success by combining the strengths of both approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05330v1">MindFlow: Revolutionizing E-commerce Customer Support with Multimodal LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled new applications in e-commerce customer service. However, their capabilities remain constrained in complex, multimodal scenarios. We present MindFlow, the first open-source multimodal LLM agent tailored for e-commerce. Built on the CoALA framework, it integrates memory, decision-making, and action modules, and adopts a modular "MLLM-as-Tool" strategy for effect visual-textual reasoning. Evaluated via online A/B testing and simulation-based ablation, MindFlow demonstrates substantial gains in handling complex queries, improving user satisfaction, and reducing operational costs, with a 93.53% relative improvement observed in real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11679v2">LLMs on support of privacy and security of mobile apps: state of the art and research directions</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 I am writing to respectfully request the withdrawal of my recent submission to arXiv due to an authorship issue. The paper was submitted without the explicit consent of two co-authors. After internal discussion, they have expressed clear disagreement with the submission and raised concerns about unresolved academic inaccuracies in the current version
    </div>
    <details class="paper-abstract">
      Modern life has witnessed the explosion of mobile devices. However, besides the valuable features that bring convenience to end users, security and privacy risks still threaten users of mobile apps. The increasing sophistication of these threats in recent years has underscored the need for more advanced and efficient detection approaches. In this chapter, we explore the application of Large Language Models (LLMs) to identify security risks and privacy violations and mitigate them for the mobile application ecosystem. By introducing state-of-the-art research that applied LLMs to mitigate the top 10 common security risks of smartphone platforms, we highlight the feasibility and potential of LLMs to replace traditional analysis methods, such as dynamic and hybrid analysis of mobile apps. As a representative example of LLM-based solutions, we present an approach to detect sensitive data leakage when users share images online, a common behavior of smartphone users nowadays. Finally, we discuss open research challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05305v1">Narrowing the Gap: Supervised Fine-Tuning of Open-Source LLMs as a Viable Alternative to Proprietary Models for Pedagogical Tools</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 7 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Frontier Large language models (LLMs) like ChatGPT and Gemini can decipher cryptic compiler errors for novice programmers, but their computational scale, cost, and tendency to over-assist make them problematic for widespread pedagogical adoption. This work demonstrates that smaller, specialised language models, enhanced via Supervised Fine-Tuning (SFT), present a more viable alternative for educational tools. We utilise a new dataset of 40,000 C compiler error explanations, derived from real introductory programming (CS1/2) student-generated programming errors, which we used to fine-tune three open-source models: Qwen3-4B, Llama-3.1-8B, and Qwen3-32B. We performed a dual evaluation, combining expert human reviews with a large-scale automated analysis of 8,000 responses using a validated LLM-as-judge ensemble. Our results show that SFT significantly boosts the pedagogical quality of smaller models, achieving performance comparable to much larger models. We analyse the trade-offs between model size and quality, confirming that fine-tuning compact, efficient models on high-quality, domain-specific data is a potent strategy for creating specialised models to drive educational tools. We provide a replicable methodology to foster broader access to generative AI capabilities in educational contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07060v1">DeepRetro: Retrosynthetic Pathway Discovery using Iterative LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 51 pages,
    </div>
    <details class="paper-abstract">
      Retrosynthesis, the identification of precursor molecules for a target compound, is pivotal for synthesizing complex molecules, but faces challenges in discovering novel pathways beyond predefined templates. Recent large language model (LLM) approaches to retrosynthesis have shown promise but effectively harnessing LLM reasoning capabilities for effective multi-step planning remains an open question. To address this challenge, we introduce DeepRetro, an open-source, iterative, hybrid LLM-based retrosynthetic framework. Our approach integrates the strengths of conventional template-based/Monte Carlo tree search tools with the generative power of LLMs in a step-wise, feedback-driven loop. Initially, synthesis planning is attempted with a template-based engine. If this fails, the LLM subsequently proposes single-step retrosynthetic disconnections. Crucially, these suggestions undergo rigorous validity, stability, and hallucination checks before the resulting precursors are recursively fed back into the pipeline for further evaluation. This iterative refinement allows for dynamic pathway exploration and correction. We demonstrate the potential of this pipeline through benchmark evaluations and case studies, showcasing its ability to identify viable and potentially novel retrosynthetic routes. In particular, we develop an interactive graphical user interface that allows expert human chemists to provide human-in-the-loop feedback to the reasoning algorithm. This approach successfully generates novel pathways for complex natural product compounds, demonstrating the potential for iterative LLM reasoning to advance state-of-art in complex chemical syntheses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05258v1">Spatio-Temporal LLM: Reasoning about Environments and Actions</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Code and data are available at https://zoezheng126.github.io/STLLM-website/
    </div>
    <details class="paper-abstract">
      Despite the significant recent progress of Multimodal Large Language Models (MLLMs), MLLMs still struggle to correctly answer prompts that require a holistic spatio-temporal understanding. Specifically, it is challenging to address prompts that refer to 1) the entirety of an environment that an agent equipped with an MLLM can operate in; and simultaneously also refer to 2) recent actions that just happened and are encoded in a video clip. However, such a holistic spatio-temporal understanding is important for agents operating in the real world. To address this issue, we first develop a framework to collect a large-scale dataset. Using the collected "Reasoning about Environments and Actions" (REA) dataset, we show that recent methods indeed struggle to correctly answer the prompts. To improve, we develop a "spatio-temporal LLM" (ST-LLM), a model equipped with projectors to improve both spatial understanding of an environment and temporal understanding of recent observations. On the collected REA data, we show that the proposed method significantly improves results compared to prior work. Code and data are available at https://zoezheng126.github.io/STLLM-website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05257v1">Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 23 Pages, Y. Hu and Y. Wang contribute equally
    </div>
    <details class="paper-abstract">
      Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. Existing datasets either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Furthermore, no existing benchmarks cover all four competencies. Therefore, we introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark combines reformulated existing datasets with newly constructed ones, covering the above four memory competencies, providing a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05606v3">OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05228v1">Cascade: Token-Sharded Private LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 To be published in ICML 2025 Main Proceedings as "Hidden No More: Attacking and Defending Private Third-Party LLM Inference", together with arXiv:2505.18332
    </div>
    <details class="paper-abstract">
      As LLMs continue to increase in parameter size, the computational resources required to run them are available to fewer parties. Therefore, third-party inference services -- where LLMs are hosted by third parties with significant computational resources -- are becoming increasingly popular. However, third party inference raises critical concerns about user data privacy. To mitigate these risks, privacy researchers have developed provably secure schemes for third-party inference, such as Secure Multi-Party Computation (SMPC). However, SMPC protocols have significant computational and communication overhead, and do not scale to large models. In this work, we propose a new multi-party inference protocol, Cascade, that avoids these punitive costs by leveraging sharding in the sequence dimension to maintain privacy, trading off cryptographic privacy guarantees for increased performance and scalability. We demonstrate that Cascade is resistant to a generalization of a recent attack that is highly effective against other statistical privacy schemes, and that it is further resistant to learning-based attacks. As Cascade is orders of magnitude faster than existing schemes, our findings offer practical solutions for secure deployment of modern state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23824v2">Reviewing Scientific Papers for Critical Problems With Reasoning LLMs: Baseline Approaches and Automatic Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Add results from new experiments; update discussion and GitHub link
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models have sparked interest in utilizing them to aid the peer review process of scientific publication amid the peer review crisis. However, having AI models generate full reviews in the same way as human reviewers risks exacerbating the irresponsible use of LLM-generated reviews. As an alternative, we propose adopting LLMs as manuscript quality checkers. We introduce several baseline approaches and an extendable automatic evaluation framework using top reasoning LLMs as judges to tackle the difficulty of recruiting domain experts for manual evaluation. Utilizing papers withdrawn from arXiv, we validated our proposed methods with several leading reasoning LLMs from multiple vendors and assessed their performance and API costs for identifying critical errors and unsoundness problems in scientific papers. o3 exhibited the best problem identification performance among all models at a modest cost. This paper provides insights into document-based scientific understanding/reasoning and lays a foundation for future applications. Our dataset, code, and model outputs are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05200v1">In-Context Learning as an Effective Estimator of Functional Correctness of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      When applying LLM-based code generation to software development projects that follow a feature-driven or rapid application development approach, it becomes necessary to estimate the functional correctness of the generated code in the absence of test cases. Just as a user selects a relevant document from a ranked list of retrieved ones, a software generation workflow requires a developer to choose (and potentially refine) a generated solution from a ranked list of alternative solutions, ordered by their posterior likelihoods. This implies that estimating the quality of a ranked list -- akin to estimating "relevance" for query performance prediction (QPP) in IR -- is also crucial for generative software development, where quality is defined in terms of "functional correctness". In this paper, we propose an in-context learning (ICL) based approach for code quality estimation. Our findings demonstrate that providing few-shot examples of functionally correct code from a training set enhances the performance of existing QPP approaches as well as a zero-shot-based approach for code quality estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05995v2">NativQA Framework: Enabling LLMs with Native, Local, and Everyday Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 LLMs, Native, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has raised concerns about cultural bias, fairness, and their applicability in diverse linguistic and underrepresented regional contexts. To enhance and benchmark the capabilities of LLMs, there is a need to develop large-scale resources focused on multilingual, local, and cultural contexts. In this study, we propose the NativQA framework, which can seamlessly construct large-scale, culturally and regionally aligned QA datasets in native languages. The framework utilizes user-defined seed queries and leverages search engines to collect location-specific, everyday information. It has been evaluated across 39 locations in 24 countries and in 7 languages -- ranging from extremely low-resource to high-resource languages -- resulting in over 300K Question-Answer (QA) pairs. The developed resources can be used for LLM benchmarking and further fine-tuning. The framework has been made publicly available for the community (https://gitlab.com/nativqa/nativqa-framework).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05118v1">VerifyLLM: LLM-Based Pre-Execution Task Plan Verification for Robots</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 IROS 2025
    </div>
    <details class="paper-abstract">
      In the field of robotics, researchers face a critical challenge in ensuring reliable and efficient task planning. Verifying high-level task plans before execution significantly reduces errors and enhance the overall performance of these systems. In this paper, we propose an architecture for automatically verifying high-level task plans before their execution in simulator or real-world environments. Leveraging Large Language Models (LLMs), our approach consists of two key steps: first, the conversion of natural language instructions into Linear Temporal Logic (LTL), followed by a comprehensive analysis of action sequences. The module uses the reasoning capabilities of the LLM to evaluate logical coherence and identify potential gaps in the plan. Rigorous testing on datasets of varying complexity demonstrates the broad applicability of the module to household tasks. We contribute to improving the reliability and efficiency of task planning and addresses the critical need for robust pre-execution verification in autonomous systems. The code is available at https://verifyllm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19794v2">Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04976v1">Can Video LLMs Refuse to Answer? Alignment for Answerability in Video Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      In the broader context of deep learning, Multimodal Large Language Models have achieved significant breakthroughs by leveraging powerful Large Language Models as a backbone to align different modalities into the language space. A prime exemplification is the development of Video Large Language Models (Video-LLMs). While numerous advancements have been proposed to enhance the video understanding capabilities of these models, they are predominantly trained on questions generated directly from video content. However, in real-world scenarios, users often pose questions that extend beyond the informational scope of the video, highlighting the need for Video-LLMs to assess the relevance of the question. We demonstrate that even the best-performing Video-LLMs fail to reject unfit questions-not necessarily due to a lack of video understanding, but because they have not been trained to identify and refuse such questions. To address this limitation, we propose alignment for answerability, a framework that equips Video-LLMs with the ability to evaluate the relevance of a question based on the input video and appropriately decline to answer when the question exceeds the scope of the video, as well as an evaluation framework with a comprehensive set of metrics designed to measure model behavior before and after alignment. Furthermore, we present a pipeline for creating a dataset specifically tailored for alignment for answerability, leveraging existing video-description paired datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04967v1">The Case for Instance-Optimized LLMs in OLAP Databases</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can enhance analytics systems with powerful data summarization, cleaning, and semantic transformation capabilities. However, deploying LLMs at scale -- processing millions to billions of rows -- remains prohibitively expensive in computation and memory. We present IOLM-DB, a novel system that makes LLM-enhanced database queries practical through query-specific model optimization. Instead of using general-purpose LLMs, IOLM-DB generates lightweight, specialized models tailored to each query's specific needs using representative data samples. IOLM-DB reduces model footprints by up to 76% and increases throughput by up to 3.31$\times$ while maintaining accuracy through aggressive compression techniques, including quantization, sparsification, and structural pruning. We further show how our approach enables higher parallelism on existing hardware and seamlessly supports caching and batching strategies to reduce overheads. Our prototype demonstrates that leveraging LLM queries inside analytics systems is feasible at scale, opening new possibilities for future OLAP applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04575v1">Lilith: Developmental Modular LLMs with Chemical Signaling</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 4 pages, 0 figures, position paper
    </div>
    <details class="paper-abstract">
      Current paradigms in Artificial Intelligence rely on layers of feedforward networks which model brain activity at the neuronal level. We conjecture that expanding to the level of multiple brain regions with chemical signaling may be a productive step toward understanding the emergence of consciousness. We propose LILITH, a novel architecture that combines developmental training of modular language models with brain-inspired token-based communication protocols, mirroring chemical signaling in the brain. Our approach models distinct brain regions as specialized LLM modules including thinking, memory, sensory, and regulatory components that communicate through emergent token-based signaling protocols analogous to neurotransmitter networks. Unlike traditional pre-trained systems, LILITH would employ developmental training where untrained LLM architectures learn through simulated life experiences, developing communication pathways and cognitive abilities through environmental interaction and evolutionary optimization. This framework would enable direct empirical investigation of consciousness emergence using Integrated Information Theory metrics while providing unprecedented insight into inter-module signaling patterns during development. By optimizing for consciousness emergence rather than task performance, LILITH could provide insight into different emergent phenomena at multiple levels of neural correlates, contrasting neuronal-level processing with multi-region coordination dynamics. The goal of this paper is to put the idea forward while recognizing the substantial challenges in implementing such a system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04562v1">Evaluating LLMs on Real-World Forecasting Against Human Superforecasters</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against human superforecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of superforecasters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17833v2">The Role of Open-Source LLMs in Shaping the Future of GeoAI</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming geospatial artificial intelligence (GeoAI), offering new capabilities in data processing, spatial analysis, and decision support. This paper examines the open-source paradigm's critical role in this transformation. While proprietary LLMs offer accessibility, they often limit the customization, interoperability, and transparency vital for specialized geospatial tasks. Conversely, open-source alternatives significantly advance Geographic Information Science (GIScience) by fostering greater adaptability, reproducibility, and community-driven innovation. Open frameworks empower researchers to tailor solutions, integrate cutting-edge methodologies (e.g., reinforcement learning, advanced spatial indexing), and align with FAIR (Findable, Accessible, Interoperable, and Reusable) principles. However, the growing reliance on any LLM necessitates careful consideration of security vulnerabilities, ethical risks, and robust governance for AI-generated geospatial outputs. This paper argues that GIScience advances best not through a single model type, but by cultivating a diverse, interoperable ecosystem combining open-source foundations for innovation, custom geospatial models, and interdisciplinary collaboration. By critically evaluating the opportunities and challenges of open-source LLMs within the broader GeoAI landscape, this work contributes to a thorough discourse on leveraging LLMs to effectively advance spatial research, policy, and decision-making in an equitable, sustainable, and scientifically rigorous manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04517v1">DOTResize: Reducing LLM Width via Discrete Optimal Transport-based Neuron Merging</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Model compression offers a promising path to reducing the cost and inaccessibility of large pre-trained models, without significantly compromising their impressive performance. Large Transformer models, including large language models (LLMs), often contain computational redundancy, which can serve as a target for new model compression methods. In this work, we specifically target neuron-level redundancies in model layers by combining groups of similar neurons into fewer neurons. We frame this width reduction as a Discrete Optimal Transport problem, and propose DOTResize, a novel Transformer compression method that uses optimal transport theory to transform and compress model weights. To ensure applicability within the Transformer architecture, we motivate and incorporate entropic regularization and matrix factorization into the transportation maps produced by our method. Unlike pruning-based approaches which discard neurons based on importance measures, DOTResize re-projects the entire neuron width, allowing the retention and redistribution of useful signal across the reduced layer. Empirical results show that compared to simple or state-of-the-art neuron width-pruning techniques, DOTResize can outperform these methods across multiple LLM families and sizes, while achieving measurable reductions in real-world computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01436v2">Challenges & Opportunities with LLM-Assisted Visualization Retargeting</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 5 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Despite the ubiquity of visualization examples published on the web, retargeting existing custom chart implementations to new datasets remains difficult, time-intensive, and tedious. The adaptation process assumes author familiarity with both the implementation of the example as well as how the new dataset might need to be transformed to fit into the example code. With recent advances in Large Language Models (LLMs), automatic adaptation of code can be achieved from high-level user prompts, reducing the barrier for visualization retargeting. To better understand how LLMs can assist retargeting and its potential limitations, we characterize and evaluate the performance of LLM assistance across multiple datasets and charts of varying complexity, categorizing failures according to type and severity. In our evaluation, we compare two approaches: (1) directly instructing the LLM model to fully generate and adapt code by treating code as text inputs and (2) a more constrained program synthesis pipeline where the LLM guides the code construction process by providing structural information (e.g., visual encodings) based on properties of the example code and data. We find that both approaches struggle when new data has not been appropriately transformed, and discuss important design recommendations for future retargeting systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04446v1">Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04444v1">Data Discovery using LLMs -- A Study of Data User Behaviour</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 Accepted as full paper at TPDL'25
    </div>
    <details class="paper-abstract">
      Data search for scientific research is more complex than a simple web search. The emergence of large language models (LLMs) and their applicability for scientific tasks offers new opportunities for researchers who are looking for data, e.g., to freely express their data needs instead of fitting them into restrictions of data catalogues and portals. However, this also creates uncertainty about whether LLMs are suitable for this task. To answer this question, we conducted a user study with 32 researchers. We qualitatively and quantitively analysed participants' information interaction behaviour while searching for data using LLMs in two data search tasks, one in which we prompted the LLM to behave as a persona. We found that participants interact with LLMs in natural language, but LLMs remain a tool for them rather than an equal conversational partner. This changes slightly when the LLM is prompted to behave as a persona, but the prompting only affects participants' user experience when they are already experienced in LLM use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04431v1">MedGellan: LLM-Generated Medical Guidance to Support Physicians</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Medical decision-making is a critical task, where errors can result in serious, potentially life-threatening consequences. While full automation remains challenging, hybrid frameworks that combine machine intelligence with human oversight offer a practical alternative. In this paper, we present MedGellan, a lightweight, annotation-free framework that uses a Large Language Model (LLM) to generate clinical guidance from raw medical records, which is then used by a physician to predict diagnoses. MedGellan uses a Bayesian-inspired prompting strategy that respects the temporal order of clinical data. Preliminary experiments show that the guidance generated by the LLM with MedGellan improves diagnostic performance, particularly in recall and $F_1$ score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02773v2">KERAP: A Knowledge-Enhanced Reasoning Approach for Accurate Zero-shot Diagnosis Prediction Using Multi-agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Medical diagnosis prediction plays a critical role in disease detection and personalized healthcare. While machine learning (ML) models have been widely adopted for this task, their reliance on supervised training limits their ability to generalize to unseen cases, particularly given the high cost of acquiring large, labeled datasets. Large language models (LLMs) have shown promise in leveraging language abilities and biomedical knowledge for diagnosis prediction. However, they often suffer from hallucinations, lack structured medical reasoning, and produce useless outputs. To address these challenges, we propose KERAP, a knowledge graph (KG)-enhanced reasoning approach that improves LLM-based diagnosis prediction through a multi-agent architecture. Our framework consists of a linkage agent for attribute mapping, a retrieval agent for structured knowledge extraction, and a prediction agent that iteratively refines diagnosis predictions. Experimental results demonstrate that KERAP enhances diagnostic reliability efficiently, offering a scalable and interpretable solution for zero-shot medical diagnosis prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09701v2">Have LLMs Made Active Learning Obsolete? Surveying the NLP Community</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Supervised learning relies on annotated data, which is expensive to obtain. A longstanding strategy to reduce annotation costs is active learning, an iterative process, in which a human annotates only data instances deemed informative by a model. Large language models (LLMs) have pushed the effectiveness of active learning, while also advancing methods such as few- or zero-shot learning, and text synthesis -- all of which can reduce the need for active learning. This naturally raises the question: has active learning become obsolete? To answer this fully, we must look beyond literature to practical experiences. We conduct an online survey in the NLP community to collect previously intangible insights on the perceived relevance of data annotation, particularly focusing on active learning, including best practices, obstacles, and future prospects. Our findings show that annotated data is expected to remain a key factor and active learning to stay highly relevant while benefiting from LLMs. Consistent with a community survey from over a decade ago, however, we find that three key challenges persist -- setup complexity, risks in the cost reduction, and tooling -- for which we propose alleviation strategies. We publish an anonymized version of the collected dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04365v1">Attention Slipping: A Mechanistic Understanding of Jailbreak Attacks and Defenses in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become more integral to society and technology, ensuring their safety becomes essential. Jailbreak attacks exploit vulnerabilities to bypass safety guardrails, posing a significant threat. However, the mechanisms enabling these attacks are not well understood. In this paper, we reveal a universal phenomenon that occurs during jailbreak attacks: Attention Slipping. During this phenomenon, the model gradually reduces the attention it allocates to unsafe requests in a user query during the attack process, ultimately causing a jailbreak. We show Attention Slipping is consistent across various jailbreak methods, including gradient-based token replacement, prompt-level template refinement, and in-context learning. Additionally, we evaluate two defenses based on query perturbation, Token Highlighter and SmoothLLM, and find they indirectly mitigate Attention Slipping, with their effectiveness positively correlated with the degree of mitigation achieved. Inspired by this finding, we propose Attention Sharpening, a new defense that directly counters Attention Slipping by sharpening the attention score distribution using temperature scaling. Experiments on four leading LLMs (Gemma2-9B-It, Llama3.1-8B-It, Qwen2.5-7B-It, Mistral-7B-It v0.2) show that our method effectively resists various jailbreak attacks while maintaining performance on benign tasks on AlpacaEval. Importantly, Attention Sharpening introduces no additional computational or memory overhead, making it an efficient and practical solution for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12440v2">HKCanto-Eval: A Benchmark for Evaluating Cantonese Language Understanding and Cultural Comprehension in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      The ability of language models to comprehend and interact in diverse linguistic and cultural landscapes is crucial. The Cantonese language used in Hong Kong presents unique challenges for natural language processing due to its rich cultural nuances and lack of dedicated evaluation datasets. The HKCanto-Eval benchmark addresses this gap by evaluating the performance of large language models (LLMs) on Cantonese language understanding tasks, extending to English and Written Chinese for cross-lingual evaluation. HKCanto-Eval integrates cultural and linguistic nuances intrinsic to Hong Kong, providing a robust framework for assessing language models in realistic scenarios. Additionally, the benchmark includes questions designed to tap into the underlying linguistic metaknowledge of the models. Our findings indicate that while proprietary models generally outperform open-weight models, significant limitations remain in handling Cantonese-specific linguistic and cultural knowledge, highlighting the need for more targeted training data and evaluation methods. The code can be accessed at https://github.com/hon9kon9ize/hkeval2025
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04315v1">HLStrans: Dataset for LLM-Driven C-to-HLS Hardware Code Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      High-level synthesis (HLS) enables software developers to describe and implement hardware at a higher level of abstraction by using C/C++ instead of traditional hardware description languages to automatically generate FPGA-ready designs. However, generating HLS code significantly differs from standard C/C++: it disallows certain coding idioms, relies on specialized libraries, and critically requires fine-grained transformations and the insertion of optimization directives (pragmas) to achieve high performance. Large language models (LLMs) have shown promise in automating such transformations, yet existing open-source datasets lack sufficient complexity and optimization diversity. To address this gap, we introduce the HLStrans dataset, a comprehensive collection of 137 distinct real word programs, each annotated with a variety of C-to-HLS transformations that yield over 23K labeled design variants. These include a broad spectrum of pragmas and code-level optimizations. We benchmark state-of-the-art LLMs on this dataset to evaluate their ability to generate synthesizable, high-performance HLS code. As part of an ongoing effort, we plan to expand the HLStrans dataset in both scale and program variety, further empowering research at the intersection of AI and hardware synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15567v2">Intelligent Assistants for the Semiconductor Failure Analysis with LLM-Based Planning Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 This technical report provides evaluation details of the epxeriments presented in the paper accepted to ISTFA 2025
    </div>
    <details class="paper-abstract">
      Failure Analysis (FA) is a highly intricate and knowledge-intensive process. The integration of AI components within the computational infrastructure of FA labs has the potential to automate a variety of tasks, including the detection of non-conformities in images, the retrieval of analogous cases from diverse data sources, and the generation of reports from annotated images. However, as the number of deployed AI models increases, the challenge lies in orchestrating these components into cohesive and efficient workflows that seamlessly integrate with the FA process. This paper investigates the design and implementation of a Large Language Model (LLM)-based Planning Agent (LPA) to assist FA engineers in solving their analysis cases. The LPA integrates LLMs with advanced planning capabilities and external tool utilization, enabling autonomous processing of complex queries, retrieval of relevant data from external systems, and generation of human-readable responses. Evaluation results demonstrate the agent's operational effectiveness and reliability in supporting FA tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04294v1">BiFair: A Fairness-aware Training Framework for LLM-enhanced Recommender Systems via Bi-level Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Large Language Model-enhanced Recommender Systems (LLM-enhanced RSs) have emerged as a powerful approach to improving recommendation quality by leveraging LLMs to generate item representations. Despite these advancements, the integration of LLMs raises severe fairness concerns. Existing studies reveal that LLM-based RSs exhibit greater unfairness than traditional RSs, yet fairness issues in LLM-enhanced RSs remain largely unexplored. In this paper, our empirical study reveals that while LLM-enhanced RSs improve fairness across item groups, a significant fairness gap persists. Further enhancement remains challenging due to the architectural differences and varying sources of unfairness inherent in LLM-enhanced RSs. To bridge this gap, we first decompose unfairness into i) \textit{prior unfairness} in LLM-generated representations and ii) \textit{training unfairness} in recommendation models. Then, we propose BiFair, a bi-level optimization-based fairness-aware training framework designed to mitigate both prior and training unfairness simultaneously. BiFair optimizes two sets of learnable parameters: LLM-generated representations and a trainable projector in the recommendation model, using a two-level nested optimization process. Additionally, we introduce an adaptive inter-group balancing mechanism, leveraging multi-objective optimization principles to dynamically balance fairness across item groups. Extensive experiments on three real-world datasets demonstrate that BiFair significantly mitigates unfairness and outperforms previous state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04295v1">LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04276v1">FIXME: Towards End-to-End Benchmarking of LLM-Aided Design Verification</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Despite the transformative potential of Large Language Models (LLMs) in hardware design, a comprehensive evaluation of their capabilities in design verification remains underexplored. Current efforts predominantly focus on RTL generation and basic debugging, overlooking the critical domain of functional verification, which is the primary bottleneck in modern design methodologies due to the rapid escalation of hardware complexity. We present FIXME, the first end-to-end, multi-model, and open-source evaluation framework for assessing LLM performance in hardware functional verification (FV) to address this crucial gap. FIXME introduces a structured three-level difficulty hierarchy spanning six verification sub-domains and 180 diverse tasks, enabling in-depth analysis across the design lifecycle. Leveraging a collaborative AI-human approach, we construct a high-quality dataset using 100% silicon-proven designs, ensuring comprehensive coverage of real-world challenges. Furthermore, we enhance the functional coverage by 45.57% through expert-guided optimization. By rigorously evaluating state-of-the-art LLMs such as GPT-4, Claude3, and LlaMA3, we identify key areas for improvement and outline promising research directions to unlock the full potential of LLM-driven automation in hardware design verification. The benchmark is available at https://github.com/ChatDesignVerification/FIXME.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06283v2">LLMs' Reading Comprehension Is Affected by Parametric Knowledge and Struggles with Hypothetical Statements</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      The task of reading comprehension (RC), often implemented as context-based question answering (QA), provides a primary means to assess language models' natural language understanding (NLU) capabilities. Yet, when applied to large language models (LLMs) with extensive built-in world knowledge, this method can be deceptive. If the context aligns with the LLMs' internal knowledge, it is hard to discern whether the models' answers stem from context comprehension or from LLMs' internal information. Conversely, using data that conflicts with the models' knowledge creates erroneous trends which distort the results. To address this issue, we suggest to use RC on imaginary data, based on fictitious facts and entities. This task is entirely independent of the models' world knowledge, enabling us to evaluate LLMs' linguistic abilities without the interference of parametric knowledge. Testing ChatGPT, GPT-4, LLaMA 2 and Mixtral on such imaginary data, we uncover a class of linguistic phenomena posing a challenge to current LLMs, involving thinking in terms of alternative, hypothetical scenarios. While all the models handle simple affirmative and negative contexts with high accuracy, they are much more prone to error when dealing with modal and conditional contexts. Crucially, these phenomena also trigger the LLMs' vulnerability to knowledge-conflicts again. In particular, while some models prove virtually unaffected by knowledge conflicts in affirmative and negative contexts, when faced with more semantically involved modal and conditional environments, they often fail to separate the text from their internal knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00025v4">Explainable AI for Mental Health Emergency Returns: Integrating LLMs with Predictive Modeling</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Importance: Emergency department (ED) returns for mental health conditions pose a major healthcare burden, with 24-27% of patients returning within 30 days. Traditional machine learning models for predicting these returns often lack interpretability for clinical use. Objective: To assess whether integrating large language models (LLMs) with machine learning improves predictive accuracy and clinical interpretability of ED mental health return risk models. Methods: This retrospective cohort study analyzed 42,464 ED visits for 27,904 unique mental health patients at an academic medical center in the Deep South from January 2018 to December 2022. Main Outcomes and Measures: Two primary outcomes were evaluated: (1) 30-day ED return prediction accuracy and (2) model interpretability using a novel LLM-enhanced framework integrating SHAP (SHapley Additive exPlanations) values with clinical knowledge. Results: For chief complaint classification, LLaMA 3 (8B) with 10-shot learning outperformed traditional models (accuracy: 0.882, F1-score: 0.86). In SDoH classification, LLM-based models achieved 0.95 accuracy and 0.96 F1-score, with Alcohol, Tobacco, and Substance Abuse performing best (F1: 0.96-0.89), while Exercise and Home Environment showed lower performance (F1: 0.70-0.67). The LLM-based interpretability framework achieved 99% accuracy in translating model predictions into clinically relevant explanations. LLM-extracted features improved XGBoost AUC from 0.74 to 0.76 and AUC-PR from 0.58 to 0.61. Conclusions and Relevance: Integrating LLMs with machine learning models yielded modest but consistent accuracy gains while significantly enhancing interpretability through automated, clinically relevant explanations. This approach provides a framework for translating predictive analytics into actionable clinical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04219v1">Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Current unlearning methods for LLMs optimize on the private information they seek to remove by incorporating it into their training objectives. We argue this not only risks reinforcing exposure to sensitive data, it also fundamentally contradicts the principle of minimizing its use. As a remedy, we propose a novel unlearning method - Partial Model Collapse (PMC), which does not require unlearning targets in the unlearning objective. Our approach is inspired by recent observations that training generative models on their own generations leads to distribution collapse, effectively removing information from the model. Our core idea is to leverage this collapse for unlearning by triggering collapse partially on the sensitive data. We theoretically analyze that our approach converges to the desired outcome, i.e. the LLM unlearns the information in the forget set. We empirically demonstrate that PMC overcomes two key limitations of existing unlearning approaches that explicitly optimize on unlearning targets, and more effectively removes private information from model outputs. Overall, our contributions represent an important step toward more comprehensive unlearning that aligns with real-world privacy constraints. Code available at https://www.cs.cit.tum.de/daml/partial-model-collapse/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03908v1">Bridging Vision and Language: Optimal Transport-Driven Radiology Report Generation via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Radiology report generation represents a significant application within medical AI, and has achieved impressive results. Concurrently, large language models (LLMs) have demonstrated remarkable performance across various domains. However, empirical validation indicates that general LLMs tend to focus more on linguistic fluency rather than clinical effectiveness, and lack the ability to effectively capture the relationship between X-ray images and their corresponding texts, thus resulting in poor clinical practicability. To address these challenges, we propose Optimal Transport-Driven Radiology Report Generation (OTDRG), a novel framework that leverages Optimal Transport (OT) to align image features with disease labels extracted from reports, effectively bridging the cross-modal gap. The core component of OTDRG is Alignment \& Fine-Tuning, where OT utilizes results from the encoding of label features and image visual features to minimize cross-modal distances, then integrating image and text features for LLMs fine-tuning. Additionally, we design a novel disease prediction module to predict disease labels contained in X-ray images during validation and testing. Evaluated on the MIMIC-CXR and IU X-Ray datasets, OTDRG achieves state-of-the-art performance in both natural language generation (NLG) and clinical efficacy (CE) metrics, delivering reports that are not only linguistically coherent but also clinically accurate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03876v1">LLMs model how humans induce logically structured rules</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      A central goal of cognitive science is to provide a computationally explicit account of both the structure of the mind and its development: what are the primitive representational building blocks of cognition, what are the rules via which those primitives combine, and where do these primitives and rules come from in the first place? A long-standing debate concerns the adequacy of artificial neural networks as computational models that can answer these questions, in particular in domains related to abstract cognitive function, such as language and logic. This paper argues that recent advances in neural networks -- specifically, the advent of large language models (LLMs) -- represent an important shift in this debate. We test a variety of LLMs on an existing experimental paradigm used for studying the induction of rules formulated over logical concepts. Across four experiments, we find converging empirical evidence that LLMs provide at least as good a fit to human behavior as models that implement a Bayesian probablistic language of thought (pLoT), which have been the best computational models of human behavior on the same task. Moreover, we show that the LLMs make qualitatively different predictions about the nature of the rules that are inferred and deployed in order to complete the task, indicating that the LLM is unlikely to be a mere implementation of the pLoT solution. Based on these results, we argue that LLMs may instantiate a novel theoretical account of the primitive representations and computations necessary to explain human logical concepts, with which future work in cognitive science should engage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03871v1">Enhancing Adaptive Behavioral Interventions with LLM Inference from Participant-Described States</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Accepted at Machine Learning for Healthcare (MLHC) 2025
    </div>
    <details class="paper-abstract">
      The use of reinforcement learning (RL) methods to support health behavior change via personalized and just-in-time adaptive interventions is of significant interest to health and behavioral science researchers focused on problems such as smoking cessation support and physical activity promotion. However, RL methods are often applied to these domains using a small collection of context variables to mitigate the significant data scarcity issues that arise from practical limitations on the design of adaptive intervention trials. In this paper, we explore an approach to significantly expanding the state space of an adaptive intervention without impacting data efficiency. The proposed approach enables intervention participants to provide natural language descriptions of aspects of their current state. It then leverages inference with pre-trained large language models (LLMs) to better align the policy of a base RL method with these state descriptions. To evaluate our method, we develop a novel physical activity intervention simulation environment that generates text-based state descriptions conditioned on latent state variables using an auxiliary LLM. We show that this approach has the potential to significantly improve the performance of online policy learning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03865v1">OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05292v1">A LLM-Driven Multi-Agent Systems for Professional Development of Mathematics Teachers</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Professional development (PD) serves as the cornerstone for teacher tutors to grasp content knowledge. However, providing equitable and timely PD opportunities for teachers poses significant challenges. To address this issue, we introduce I-VIP (Intelligent Virtual Interactive Program), an intelligent tutoring platform for teacher professional development, driven by large language models (LLMs) and supported by multi-agent frameworks. This platform offers a user-friendly conversational interface and allows users to employ a variety of interactive tools to facilitate question answering, knowledge comprehension, and reflective summarization while engaging in dialogue. To underpin the functionality of this platform, including knowledge expectation analysis, response scoring and classification, and feedback generation, the multi-agent frameworks are leveraged to enhance the accuracy of judgments and mitigate the issue of missing key points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16429v2">The Transformative Influence of LLMs on Software Development & Developer Productivity</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 10 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      The increasing adoption and commercialization of generalized Large Language Models (LLMs) have profoundly impacted various aspects of our daily lives. Initially embraced by the computer science community, the versatility of LLMs has found its way into diverse domains. In particular, the software engineering realm has witnessed the most transformative changes. With LLMs increasingly serving as AI Pair Programming Assistants spurred the development of specialized models aimed at aiding software engineers. Although this new paradigm offers numerous advantages, it also presents critical challenges and open problems. To identify the potential and prevailing obstacles, we systematically reviewed contemporary scholarly publications, emphasizing the perspectives of software developers and usability concerns. Preliminary findings underscore pressing concerns about data privacy, bias, and misinformation. Additionally, we identified several usability challenges, including prompt engineering, increased cognitive demands, and mistrust. Finally, we introduce 12 open problems that we have identified through our survey, covering these various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04132v1">An HTR-LLM Workflow for High-Accuracy Transcription and Analysis of Abbreviated Latin Court Hand</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      This article presents and validates an ideal, four-stage workflow for the high-accuracy transcription and analysis of challenging medieval legal documents. The process begins with a specialized Handwritten Text Recognition (HTR) model, itself created using a novel "Clean Ground Truth" curation method where a Large Language Model (LLM) refines the training data. This HTR model provides a robust baseline transcription (Stage 1). In Stage 2, this baseline is fed, along with the original document image, to an LLM for multimodal post-correction, grounding the LLM's analysis and improving accuracy. The corrected, abbreviated text is then expanded into full, scholarly Latin using a prompt-guided LLM (Stage 3). A final LLM pass performs Named-Entity Correction (NEC), regularizing proper nouns and generating plausible alternatives for ambiguous readings (Stage 4). We validate this workflow through detailed case studies, achieving Word Error Rates (WER) in the range of 2-7% against scholarly ground truths. The results demonstrate that this hybrid, multi-stage approach effectively automates the most laborious aspects of transcription while producing a high-quality, analyzable output, representing a powerful and practical solution for the current technological landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04105v1">Enhancing Robustness of LLM-Driven Multi-Agent Systems through Randomized Smoothing</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Preprint accepted by Chinese Journal of Aeronautics
    </div>
    <details class="paper-abstract">
      This paper presents a defense framework for enhancing the safety of large language model (LLM) empowered multi-agent systems (MAS) in safety-critical domains such as aerospace. We apply randomized smoothing, a statistical robustness certification technique, to the MAS consensus context, enabling probabilistic guarantees on agent decisions under adversarial influence. Unlike traditional verification methods, our approach operates in black-box settings and employs a two-stage adaptive sampling mechanism to balance robustness and computational efficiency. Simulation results demonstrate that our method effectively prevents the propagation of adversarial behaviors and hallucinations while maintaining consensus performance. This work provides a practical and scalable path toward safe deployment of LLM-based MAS in real-world, high-stakes environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04103v1">How to Train Your LLM Web Agent: A Statistical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14492v2">FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to capturing biases from training corpus, leading to potential negative social impacts. Existing prompt-based debiasing methods exhibit instability due to their sensitivity to prompt changes, while fine-tuning-based techniques incur substantial computational overhead and catastrophic forgetting. In this paper, we propose FairSteer, a novel inference-time debiasing framework without requiring customized prompt design or model retraining. Motivated by the linear representation hypothesis, our preliminary investigation demonstrates that fairness-related features can be encoded into separable directions in the hidden activation space. FairSteer operates in three steps: biased activation detection, debiasing steering vector (DSV) computation, and dynamic activation steering. Specifically, it first trains a lightweight linear classifier to detect bias signatures in activations, and then computes DSVs as intervention directions derived from small contrastive prompt pairs. Subsequently, it performs debiasing by adjusting activations with DSVs in the inference stage. Comprehensive evaluation with six LLMs demonstrates the superiority of FairSteer across question-answering, counterfactual input evaluation and open-ended text generation tasks. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04055v1">Rethinking and Exploring String-Based Malware Family Classification in the Era of LLMs and RAG</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Malware Family Classification (MFC) aims to identify the fine-grained family (e.g., GuLoader or BitRAT) to which a potential malware sample belongs, in contrast to malware detection or sample classification that predicts only an Yes/No. Accurate family identification can greatly facilitate automated sample labeling and understanding on crowdsourced malware analysis platforms such as VirusTotal and MalwareBazaar, which generate vast amounts of data daily. In this paper, we explore and assess the feasibility of using traditional binary string features for MFC in the new era of large language models (LLMs) and Retrieval-Augmented Generation (RAG). Specifically, we investigate how Family-Specific String (FSS) features could be utilized in a manner similar to RAG to facilitate MFC. To this end, we develop a curated evaluation framework covering 4,347 samples from 67 malware families, extract and analyze over 25 million strings, and conduct detailed ablation studies to assess the impact of different design choices in four major modules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00225v3">Large-scale, Independent and Comprehensive study of the power of LLMs for test case generation</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Unit testing is essential for software reliability, yet manual test creation is time-consuming and often neglected. Although search-based software testing improves efficiency, it produces tests with poor readability and maintainability. Although LLMs show promise for test generation, existing research lacks comprehensive evaluation across execution-driven assessment, reasoning-based prompting, and real-world testing scenarios. This study presents the first large-scale empirical evaluation of LLM-generated unit tests at the class level, systematically analyzing four state-of-the-art models - GPT-3.5, GPT-4, Mistral 7B, and Mixtral 8x7B - against EvoSuite across 216,300 test cases from Defects4J, SF110, and CMD (a dataset mitigating LLM training data leakage). We evaluate five prompting techniques - Zero-Shot Learning (ZSL), Few-Shot Learning (FSL), Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Guided Tree-of-Thought (GToT) - assessing syntactic correctness, compilability, hallucination-driven failures, readability, code coverage metrics, and fault detection capabilities. Our findings challenge prior claims that in-context learning is ineffective for test generation in code-specialized LLMs. Reasoning-based prompting - particularly GToT - significantly enhances test reliability, compilability, and structural adherence in general-purpose LLMs. However, hallucination-driven failures remain a persistent challenge, manifesting as non-existent symbol references, incorrect API calls, and fabricated dependencies, resulting in high compilation failure rates (up to 86%). Execution-based classification and mutation testing reveal that many failing tests stem from hallucinated dependencies, limiting effective fault detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04034v1">Lyria: A General LLM-Driven Genetic Algorithm Framework for Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive abilities across various domains, they still struggle with complex problems characterized by multi-objective optimization, precise constraint satisfaction, immense solution spaces, etc. To address the limitation, drawing on the superior semantic understanding ability of LLMs and also the outstanding global search and optimization capability of genetic algorithms, we propose to capitalize on their respective strengths and introduce Lyria, a general LLM-driven genetic algorithm framework, comprising 7 essential components. Through conducting extensive experiments with 4 LLMs across 3 types of problems, we demonstrated the efficacy of Lyria. Additionally, with 7 additional ablation experiments, we further systematically analyzed and elucidated the factors that affect its performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04009v1">Easy Dataset: A Unified and Extensible Framework for Synthesizing LLM Fine-Tuning Data from Unstructured Documents</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance on general-purpose tasks, yet adapting them to specific domains remains challenging due to the scarcity of high-quality domain data. Existing data synthesis tools often struggle to extract reliable fine-tuning data from heterogeneous documents effectively. To address this limitation, we propose Easy Dataset, a unified framework for synthesizing fine-tuning data from unstructured documents via an intuitive graphical user interface (GUI). Specifically, Easy Dataset allows users to easily configure text extraction models and chunking strategies to transform raw documents into coherent text chunks. It then leverages a persona-driven prompting approach to generate diverse question-answer pairs using public-available LLMs. Throughout the pipeline, a human-in-the-loop visual interface facilitates the review and refinement of intermediate outputs to ensure data quality. Experiments on a financial question-answering task show that fine-tuning LLMs on the synthesized dataset significantly improves domain-specific performance while preserving general knowledge. The source code and installable package are available at https://github.com/ConardLi/easy-dataset and have garnered over 9,000 GitHub stars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14363v2">Improving RL Exploration for LLM Reasoning through Retrospective Replay</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has increasingly become a pivotal technique in the post-training of large language models (LLMs). The effective exploration of the output space is essential for the success of RL. We observe that for complex problems, during the early stages of training, the model exhibits strong exploratory capabilities and can identify promising solution ideas. However, its limited capability at this stage prevents it from successfully solving these problems. The early suppression of these potentially valuable solution ideas by the policy gradient hinders the model's ability to revisit and re-explore these ideas later. Consequently, although the LLM's capabilities improve in the later stages of training, it still struggles to effectively address these complex problems. To address this exploration issue, we propose a novel algorithm named Retrospective Replay-based Reinforcement Learning (RRL), which introduces a dynamic replay mechanism throughout the training process. RRL enables the model to revisit promising states identified in the early stages, thereby improving its efficiency and effectiveness in exploration. To evaluate the effectiveness of RRL, we conduct extensive experiments on complex reasoning tasks, including mathematical reasoning and code generation, and general dialogue tasks. The results indicate that RRL maintains high exploration efficiency throughout the training period, significantly enhancing the effectiveness of RL in optimizing LLMs for complicated reasoning tasks. Moreover, it also improves the performance of RLHF, making the model both safer and more helpful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03958v1">A Comparative Study of Specialized LLMs as Dense Retrievers</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Accepted by CCIR25 and published by Springer LNCS or LNAI
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly deployed as dense retrievers, the impact of their domain-specific specialization on retrieval effectiveness remains underexplored. This investigation systematically examines how task-specific adaptations in LLMs influence their retrieval capabilities, an essential step toward developing unified retrievers capable of handling text, code, images, and multimodal content. We conduct extensive experiments with eight Qwen2.5 7B LLMs, including base, instruction-tuned, code/math-specialized, long reasoning, and vision-language models across zero-shot retrieval settings and the supervised setting. For the zero-shot retrieval settings, we consider text retrieval from the BEIR benchmark and code retrieval from the CoIR benchmark. Further, to evaluate supervised performance, all LLMs are fine-tuned on the MS MARCO dataset. We find that mathematical specialization and the long reasoning capability cause consistent degradation in three settings, indicating conflicts between mathematical reasoning and semantic matching. The vision-language model and code-specialized LLMs demonstrate superior zero-shot performance compared to other LLMs, even surpassing BM25 on the code retrieval task, and maintain comparable performance to base LLMs in supervised settings. These findings suggest promising directions for the unified retrieval task leveraging cross-domain and cross-modal fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03945v1">Function-based Labels for Complementary Recommendation: Definition, Annotation, and LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Complementary recommendations enhance the user experience by suggesting items that are frequently purchased together while serving different functions from the query item. Inferring or evaluating whether two items have a complementary relationship requires complementary relationship labels; however, defining these labels is challenging because of the inherent ambiguity of such relationships. Complementary labels based on user historical behavior logs attempt to capture these relationships, but often produce inconsistent and unreliable results. Recent efforts have introduced large language models (LLMs) to infer these relationships. However, these approaches provide a binary classification without a nuanced understanding of complementary relationships. In this study, we address these challenges by introducing Function-Based Labels (FBLs), a novel definition of complementary relationships independent of user purchase logs and the opaque decision processes of LLMs. We constructed a human-annotated FBLs dataset comprising 2,759 item pairs and demonstrated that it covered possible item relationships and minimized ambiguity. We then evaluated whether some machine learning (ML) methods using annotated FBLs could accurately infer labels for unseen item pairs, and whether LLM-generated complementary labels align with human perception. Our results demonstrate that even with limited data, ML models, such as logistic regression and SVM achieve high macro-F1 scores (approximately 0.82). Furthermore, LLMs, such as gpt-4o-mini, demonstrated high consistency (0.989) and classification accuracy (0.849) under the detailed definition of FBLs, indicating their potential as effective annotators that mimic human judgment. Overall, our study presents FBLs as a clear definition of complementary relationships, enabling more accurate inferences and automated labeling of complementary recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03834v1">Economic Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Practitioners often navigate LLM performance trade-offs by plotting Pareto frontiers of optimal accuracy-cost trade-offs. However, this approach offers no way to compare between LLMs with distinct strengths and weaknesses: for example, a cheap, error-prone model vs a pricey but accurate one. To address this gap, we propose economic evaluation of LLMs. Our framework quantifies the performance trade-off of an LLM as a single number based on the economic constraints of a concrete use case, all expressed in dollars: the cost of making a mistake, the cost of incremental latency, and the cost of abstaining from a query. We apply our economic evaluation framework to compare the performance of reasoning and non-reasoning models on difficult questions from the MATH benchmark, discovering that reasoning models offer better accuracy-cost tradeoffs as soon as the economic cost of a mistake exceeds \$0.01. In addition, we find that single large LLMs often outperform cascades when the cost of making a mistake is as low as \$0.1. Overall, our findings suggest that when automating meaningful human tasks with AI models, practitioners should typically use the most powerful available model, rather than attempt to minimize AI deployment costs, since deployment costs are likely dwarfed by the economic impact of AI errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03829v1">RELRaE: LLM-Based Relationship Extraction, Labelling, Refinement, and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 18 Pages, 8 Tables, Under-review at ISWC 2025
    </div>
    <details class="paper-abstract">
      A large volume of XML data is produced in experiments carried out by robots in laboratories. In order to support the interoperability of data between labs, there is a motivation to translate the XML data into a knowledge graph. A key stage of this process is the enrichment of the XML schema to lay the foundation of an ontology schema. To achieve this, we present the RELRaE framework, a framework that employs large language models in different stages to extract and accurately label the relationships implicitly present in the XML schema. We investigate the capability of LLMs to accurately generate these labels and then evaluate them. Our work demonstrates that LLMs can be effectively used to support the generation of relationship labels in the context of lab automation, and that they can play a valuable role within semi-automatic ontology generation frameworks more generally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03722v1">Roadmap for using large language models (LLMs) to accelerate cross-disciplinary research with an example from computational biology</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are powerful artificial intelligence (AI) tools transforming how research is conducted. However, their use in research has been met with skepticism, due to concerns about hallucinations, biases and potential harms to research. These emphasize the importance of clearly understanding the strengths and weaknesses of LLMs to ensure their effective and responsible use. Here, we present a roadmap for integrating LLMs into cross-disciplinary research, where effective communication, knowledge transfer and collaboration across diverse fields are essential but often challenging. We examine the capabilities and limitations of LLMs and provide a detailed computational biology case study (on modeling HIV rebound dynamics) demonstrating how iterative interactions with an LLM (ChatGPT) can facilitate interdisciplinary collaboration and research. We argue that LLMs are best used as augmentative tools within a human-in-the-loop framework. Looking forward, we envisage that the responsible use of LLMs will enhance innovative cross-disciplinary research and substantially accelerate scientific discoveries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03711v1">Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted paper at MAPR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, \^O \u{A}n Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the \^O \u{A}n Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03662v1">Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Recent work has shown that fine-tuning large language models (LLMs) on code with security vulnerabilities can result in misaligned and unsafe behaviors across broad domains. These results prompted concerns about the emergence of harmful behaviors from narrow domain fine-tuning. In this paper, we contextualize these findings by analyzing how such narrow adaptation impacts the internal mechanisms and behavioral manifestations of LLMs. Through a series of experiments covering output probability distributions, loss and gradient vector geometry, layer-wise activation dynamics, and activation space dimensions, we find that behaviors attributed to "emergent misalignment" may be better interpreted as an erosion of prior alignment. We show that fine tuning on insecure code induces internal changes that oppose alignment. Further, we identify a shared latent dimension in the model's activation space that governs alignment behavior. We show that this space is activated by insecure code and by misaligned responses more generally, revealing how narrow fine-tuning can degrade general safety behavior by interfering with shared internal mechanisms. Our findings offer a mechanistic interpretation for previously observed misalignment phenomena, and highlights the fragility of alignment in LLMs. The results underscore the need for more robust fine-tuning strategies that preserve intended behavior across domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03659v1">Specification-Guided Repair of Arithmetic Errors in Dafny Programs using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Formal verification offers strong assurances of software correctness. However, debugging and repairing the underlying faults can be complex and time-consuming when verification fails. Automated Program Repair (APR) aims to ease this by automatically identifying and fixing faults. Traditional APR techniques often depend on test suites for validation, but these may fail to capture all scenarios. In contrast, formal specifications provide stronger correctness criteria for effective repairs. We present an innovative APR tool for Dafny, a verification-aware programming language that uses formal specifications - including pre-conditions, post-conditions, and invariants - as oracles for fault localization and repair. Assuming the correctness of the specifications and focusing on arithmetic bugs, we localize faults through a series of steps, which include using Hoare Logic to determine the state of each statement within the program and state-of-the-art Large Language Models (LLMs) to synthesize candidate fixes. The chosen models were GPT-4o mini, Llama 3, Mistral 7B, and Llemma 7B. We evaluate our approach using DafnyBench, a benchmark of real-world Dafny programs. Our tool achieves 89.6% accuracy in fault localization, with GPT-4o mini yielding the highest repair success rate (74.18%). These results highlight the potential of combining formal reasoning with LLM-driven program synthesis for automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08276v2">Exploring Robustness of LLMs to Paraphrasing Based on Sociodemographic Factors</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Despite their linguistic prowess, LLMs have been shown to be vulnerable to small input perturbations. While robustness to local adversarial changes has been studied, robustness to global modifications such as different linguistic styles remains underexplored. Therefore, we take a broader approach to explore a wider range of variations across sociodemographic dimensions. We extend the SocialIQA dataset to create diverse paraphrased sets conditioned on sociodemographic factors (age and gender). The assessment aims to provide a deeper understanding of LLMs in (a) their capability of generating demographic paraphrases with engineered prompts and (b) their capabilities in interpreting real-world, complex language scenarios. We also perform a reliability analysis of the generated paraphrases looking into linguistic diversity and perplexity as well as manual evaluation. We find that demographic-based paraphrasing significantly impacts the performance of language models, indicating that the subtleties of linguistic variation remain a significant challenge. We will make the code and dataset available for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08203v2">ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted to LLMSEC Workshop at ACL 2025
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown impressive capabilities in math problem-solving tasks, their robustness to noisy inputs is not well-studied. We propose ArithmAttack to examine how robust the LLMs are when they encounter noisy prompts that contain extra noise in the form of punctuation marks. While being easy to implement, ArithmAttack does not cause any information loss since words are not added or deleted from the context. We evaluate the robustness of eight LLMs, including LLama3, Mistral, Mathstral, and DeepSeek on noisy GSM8K and MultiArith datasets. Our experiments suggest that all the studied models show vulnerability to such noise, with more noise leading to poorer performances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09066v2">Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03619v1">Blackbox Dataset Inference for LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs. In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19937v2">ALAS: Measuring Latent Speech-Text Alignment For Spoken Language Understanding In Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in Spoken Language Understanding (SLU), where effective multimodal learning depends on the alignment between audio and text. Despite various fusion methods, no standard metric exists to assess this alignment. This work introduces ALAS (Automatic Latent Alignment Score), a metric that evaluates alignment by measuring correlations between audio and text representations across transformer layers. Experiments on Spoken Question Answering and Emotion Recognition show that ALAS captures meaningful patterns across tasks and layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03605v1">Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      We investigate the behaviour space of meta-heuristic optimisation algorithms automatically generated by Large Language Model driven algorithm discovery methods. Using the Large Language Evolutionary Algorithm (LLaMEA) framework with a GPT o4-mini LLM, we iteratively evolve black-box optimisation heuristics, evaluated on 10 functions from the BBOB benchmark suite. Six LLaMEA variants, featuring different mutation prompt strategies, are compared and analysed. We log dynamic behavioural metrics including exploration, exploitation, convergence and stagnation measures, for each run, and analyse these via visual projections and network-based representations. Our analysis combines behaviour-based projections, Code Evolution Graphs built from static code features, performance convergence curves, and behaviour-based Search Trajectory Networks. The results reveal clear differences in search dynamics and algorithm structures across LLaMEA configurations. Notably, the variant that employs both a code simplification prompt and a random perturbation prompt in a 1+1 elitist evolution strategy, achieved the best performance, with the highest Area Over the Convergence Curve. Behaviour-space visualisations show that higher-performing algorithms exhibit more intensive exploitation behaviour and faster convergence with less stagnation. Our findings demonstrate how behaviour-space analysis can explain why certain LLM-designed heuristics outperform others and how LLM-driven algorithm discovery navigates the open-ended and complex search space of algorithms. These findings provide insights to guide the future design of adaptive LLM-driven algorithm generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03378v1">Making Sense of Korean Sentences: A Comprehensive Evaluation of LLMs through KoSEnd Dataset</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 ACL 2025 student research workshop
    </div>
    <details class="paper-abstract">
      Although LLMs have made significant progress in various languages, there are still concerns about their effectiveness with low-resource agglutinative languages compared to languages such as English. In this study, we focused on Korean, a language known for its complex sentence endings, and evaluated LLMs on this challenging aspect. We introduce the Korean Sentence Endings (KoSEnd) dataset, which includes 3,000 sentences, each annotated for the naturalness of 15 sentence ending forms. These were collected from diverse sources to cover a range of contexts. We evaluated 11 LLMs to assess their understanding of Korean sentence endings, analyzing them based on parameter count and prediction consistency. Notably, we found that informing models about the possibility of missing sentence endings improved performance, highlighting the impact of explicitly considering certain linguistic features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03336v1">Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03327v1">Read Quietly, Think Aloud: Decoupling Comprehension and Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding text and generating high-quality responses. However, a critical distinction from human cognition is their typical lack of a distinct internal `reading' or deliberation phase before `speaking' (i.e., generating text). Humans often engage in silent reading to comprehend context and formulate thoughts prior to articulation. This paper investigates methods to imbue LLMs with a similar capacity for internal processing. We introduce and evaluate techniques that encourage LLMs to `read silently.' Our findings indicate that even a straightforward approach, such as providing the model with an initial contextual prompt or `reading space' before it begins predicting subsequent tokens for the final output, can yield significant performance improvements. We further enhance this concept by developing a `reading buddy' architecture, where an auxiliary component silently processes the input and provides refined contextual insights to the primary generation model. These approaches aim to foster deeper understanding from LLMs so that they can produce better reasoned responses, moving them one step closer to more human-like text processing. Our results indicate that these simple techniques can provide surprisingly strong impact on accuracy with multiple point accuracy boost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03326v1">Mirror in the Model: Ad Banner Image Generation via Reflective Multi-LLM and Multi-modal Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Recent generative models such as GPT-4o have shown strong capabilities in producing high-quality images with accurate text rendering. However, commercial design tasks like advertising banners demand more than visual fidelity -- they require structured layouts, precise typography, consistent branding, and more. In this paper, we introduce MIMO (Mirror In-the-Model), an agentic refinement framework for automatic ad banner generation. MIMO combines a hierarchical multi-modal agent system (MIMO-Core) with a coordination loop (MIMO-Loop) that explores multiple stylistic directions and iteratively improves design quality. Requiring only a simple natural language based prompt and logo image as input, MIMO automatically detects and corrects multiple types of errors during generation. Experiments show that MIMO significantly outperforms existing diffusion and LLM-based baselines in real-world banner design scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02881v3">Rewriting Pre-Training Data Boosts LLM Performance in Math and Code</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03305v1">Analysis and Optimized CXL-Attached Memory Allocation for Long-Context LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 9 pages, 10 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The growing prevalence of Large Language Models (LLMs) and their substantial memory requirements have prompted renewed interest in CPU offloading as a method to compensate for limited GPU memory. In particular, when CPU memory is leveraged to temporarily store intermediate states of LLMs, CPU memory becomes a new bottleneck and soon reaches the capacity limitation of commodity CPUs. In this work, we investigate the effectiveness of Compute Express Link (CXL) add-in card (AIC) memory as an extension to CPU memory, enabling larger model sizes and longer context lengths during fine-tuning. Through extensive benchmarking, this study quantifies the performance overhead introduced by transferring data between CXL memory, CPU, and GPUs, focusing on how concurrency and data volume influence bandwidth utilization and latency. This study also compares CPUbased optimizer steps when model parameters, gradients, and optimizer states reside in local memory versus CXL memory, revealing that naive adoption of CXL often degrades performance during the optimizer phase. To overcome these challenges, this study proposes a CXL-aware allocation to strategically partition CPU offloading workloads across both local and CXL memory. This study further demonstrates that employing multiple AICs significantly reduces bandwidth contention, thus improving scalability. Experimental results show that these optimizations enable efficient long-context LLM fine-tuning, underscoring CXL as a promising avenue for unlocking the full potential of CPU offloading in long-context LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03294v1">MGAA: Multi-Granular Adaptive Allocation fof Low-Rank Compression of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The enormous parameter scale of large language models (LLMs) has made model compression a research hotspot, which aims to alleviate computational resource demands during deployment and inference. As a promising direction, low-rank approximation technique has made remarkable achievements. Nevertheless, unfortunately, the vast majority of studies to low-rank approximation compression generally apply uniform compression ratios across all weight matrices, while disregarding their inherently differentiated impacts on the model's performance. Although a few recent work attempts to employ heuristic search strategies to achieve the optimal parameter allocation, such strategies are computationally inefficient and lose the generalization ability in the era of LLMs. In this study, we propose a novel parameter Multi-Granular Adaptive Allocation (MGAA) method, which can adaptively allocate parameters between and within sublayers without task-specific evaluations in the compression process. MGAA consists of two components: 1) Among different sublayers, it assigns compression ratios based on their cosine similarity between inputs and outputs, allowing for a more tailored compression in sublayers with varying degrees of importance, and 2) Within each sublayer, it allocates different compression ratios to weight matrices based on their energy distribution characteristics, ensuring a consistent energy retention ratio while optimizing compression efficiency. Comprehensive evaluations of MGAA across multiple LLMs backbone models and benchmark datasets demonstrate its superior performance. Additionally, we apply our MGAA to multimodal model LLaVA, exhibiting remarkable performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03293v1">LTLCrit: A Temporal Logic-based LLM Critic for Safe and Efficient Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promise in reasoning tasks and general decision-making in static environments. In long-term planning tasks, however, errors tend to accumulate, often leading to unsafe or inefficient behavior, limiting their use in general-purpose settings. We propose a modular actor-critic architecture in which an LLM actor is guided by LTLCrit, a trajectory-level LLM critic that communicates via linear temporal logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. The architecture supports both fixed, hand-specified safety constraints and adaptive, learned soft constraints that promote long-term efficiency. Our architecture is model-agnostic: any LLM-based planner can serve as the actor, and LTLCrit serves as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LTLCrit to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. We evaluate our system on the Minecraft diamond-mining benchmark, achieving 100% completion rates and improving efficiency compared to baseline LLM planners. Our results suggest that enabling LLMs to supervise each other through logic is a powerful and flexible paradigm for safe, generalizable decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04428v2">MoralBench: Moral Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted to ACM SIGKDD Explorations Volume 27 Issue 1
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of artificial intelligence, large language models (LLMs) have emerged as powerful tools for a myriad of applications, from natural language processing to decision-making support systems. However, as these models become increasingly integrated into societal frameworks, the imperative to ensure they operate within ethical and moral boundaries has never been more critical. This paper introduces a novel benchmark designed to measure and compare the moral reasoning capabilities of LLMs. We present the first comprehensive dataset specifically curated to probe the moral dimensions of LLM outputs, addressing a wide range of ethical dilemmas and scenarios reflective of real-world complexities. The main contribution of this work lies in the development of benchmark datasets and metrics for assessing the moral identity of LLMs, which accounts for nuance, contextual sensitivity, and alignment with human ethical standards. Our methodology involves a multi-faceted approach, combining quantitative analysis with qualitative insights from ethics scholars to ensure a thorough evaluation of model performance. By applying our benchmark across several leading LLMs, we uncover significant variations in moral reasoning capabilities of different models. These findings highlight the importance of considering moral reasoning in the development and evaluation of LLMs, as well as the need for ongoing research to address the biases and limitations uncovered in our study. We publicly release the benchmark at https://drive.google.com/drive/u/0/folders/1k93YZJserYc2CkqP8d4B3M3sgd3kA8W7 and also open-source the code of the project at https://github.com/agiresearch/MoralBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03254v1">CodeAgents: A Token-Efficient Framework for Codified Multi-Agent Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Effective prompt design is essential for improving the planning capabilities of large language model (LLM)-driven agents. However, existing structured prompting strategies are typically limited to single-agent, plan-only settings, and often evaluate performance solely based on task accuracy - overlooking critical factors such as token efficiency, modularity, and scalability in multi-agent environments. To address these limitations, we introduce CodeAgents, a prompting framework that codifies multi-agent reasoning and enables structured, token-efficient planning in multi-agent systems. In CodeAgents, all components of agent interaction - Task, Plan, Feedback, system roles, and external tool invocations - are codified into modular pseudocode enriched with control structures (e.g., loops, conditionals), boolean logic, and typed variables. This design transforms loosely connected agent plans into cohesive, interpretable, and verifiable multi-agent reasoning programs. We evaluate the proposed framework across three diverse benchmarks - GAIA, HotpotQA, and VirtualHome - using a range of representative LLMs. Results show consistent improvements in planning performance, with absolute gains of 3-36 percentage points over natural language prompting baselines. On VirtualHome, our method achieves a new state-of-the-art success rate of 56%. In addition, our approach reduces input and output token usage by 55-87% and 41-70%, respectively, underscoring the importance of token-aware evaluation metrics in the development of scalable multi-agent LLM systems. The code and resources are available at: https://anonymous.4open.science/r/CodifyingAgent-5A86
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10401v2">HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      The rapid growth of deep learning has driven exponential increases in model parameters and computational demands. NVIDIA GPUs and their CUDA-based software ecosystem provide robust support for parallel computing, significantly alleviating computational bottlenecks. Meanwhile, due to the cultivation of user programming habits and the high performance of GPUs, the CUDA ecosystem has established a dominant position in the field of parallel software. This dominance requires other hardware platforms to support CUDA-based software with performance portability. However, translating CUDA code to other platforms poses significant challenges due to differences in parallel programming paradigms and hardware architectures. Existing approaches rely on language extensions, domain-specific languages (DSLs), or compilers but face limitations in workload coverage and generalizability. Moreover, these methods often incur substantial development costs. Recently, LLMs have demonstrated extraordinary potential in various vertical domains, especially in code-related tasks. However, the performance of existing LLMs in CUDA transpilation, particularly for high-performance code, remains suboptimal. To address these challenges, we propose a novel framework for generating high-performance CUDA and corresponding platform code pairs, leveraging AI compiler and automatic optimization technology. We further enhance the framework with a graph-based data augmentation method and introduce HPCTransEval, a benchmark for evaluating LLM performance on CUDA transpilation. We conduct experiments using CUDA-to-CPU transpilation as a case study on leading LLMs. The speedup ratio of the CPU operators has an average improvemnet of 43.8\%, highlighting the potential of LLMs to address compatibility challenges within the CUDA ecosystem. Our code is available at https://github.com/PJLAB-CHIP/HPCTransCompile.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06845v6">7B Fully Open Source Moxin-LLM/VLM -- From Pretraining to GRPO-based Reinforcement Learning Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Although open-source LLMs present unprecedented opportunities for innovation and research, the commercialization of LLMs has raised concerns about transparency, reproducibility, and safety. Many open-source LLMs fail to meet fundamental transparency requirements by withholding essential components like training code and data, which may hinder further innovations on LLMs. To mitigate this issue, we introduce Moxin 7B, a fully open-source LLM developed, adhering to principles of open science, open source, open data, and open access. We release the pre-training code and configurations, training and fine-tuning datasets, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. After pre-training the base model, we finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our Instruct model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization (GRPO) following DeepSeek R1 to finetune our model, leading to the Moxin Reasoning model. Moreover, we develop our vision language model based on our Moxin model. Experiments show that our models achieve superior performance in various evaluations such as zero-shot evaluation, few-shot evaluation, and CoT evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05282v1">Exploring LLM Capabilities in Extracting DCAT-Compatible Metadata for Data Cataloging</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Efficient data exploration is crucial as data becomes increasingly important for accelerating processes, improving forecasts and developing new business models. Data consumers often spend 25-98 % of their time searching for suitable data due to the exponential growth, heterogeneity and distribution of data. Data catalogs can support and accelerate data exploration by using metadata to answer user queries. However, as metadata creation and maintenance is often a manual process, it is time-consuming and requires expertise. This study investigates whether LLMs can automate metadata maintenance of text-based data and generate high-quality DCAT-compatible metadata. We tested zero-shot and few-shot prompting strategies with LLMs from different vendors for generating metadata such as titles and keywords, along with a fine-tuned model for classification. Our results show that LLMs can generate metadata comparable to human-created content, particularly on tasks that require advanced semantic understanding. Larger models outperformed smaller ones, and fine-tuning significantly improves classification accuracy, while few-shot prompting yields better results in most cases. Although LLMs offer a faster and reliable way to create metadata, a successful application requires careful consideration of task-specific criteria and domain context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05279v1">ReservoirChat: Interactive Documentation Enhanced with LLM and Knowledge Graph for ReservoirPy</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      We introduce a tool designed to improve the capabilities of Large Language Models (LLMs) in assisting with code development using the ReservoirPy library, as well as in answering complex questions in the field of Reservoir Computing. By incorporating external knowledge through Retrieval-Augmented Generation (RAG) and knowledge graphs, our approach aims to reduce hallucinations and increase the factual accuracy of generated responses. The system provides an interactive experience similar to ChatGPT, tailored specifically for ReservoirPy, enabling users to write, debug, and understand Python code while accessing reliable domain-specific insights. In our evaluation, while proprietary models such as ChatGPT-4o and NotebookLM performed slightly better on general knowledge questions, our model outperformed them on coding tasks and showed a significant improvement over its base model, Codestral-22B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22694v2">VOCABTRIM: Vocabulary Pruning for Efficient Speculative Decoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 8 pages, 4 figures, 5 tables, accepted at ICML 2025 workshop on Efficient Systems for Foundational Models
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a simple training-free technique to improve the performance of drafter-based speculative decoding (SpD) methods that incorporates language modeling head (LM head) during drafting process. A drafter-based speculative decoding leverages one or more smaller language models, a.k.a. drafters or draft models, to sample a draft sequence or tree consisting of multiple tokens, followed by verification by a base LLM, a target model, accepting a subset as its valid generation. As it is usually considered that the speculative decoding requires one-to-one mapping between vocabularies of the target model and the draft model, it has been natural to share the vocabulary between them, or even share the LM head as in EAGLE or Medusa. We first identify that this draft token sampling scheme inherently contains an unnecessary inference overhead in drafting, especially for some target LLMs with very large vocabularies. Then, we propose a simple technique, VocabTrim, to mitigate the drafting overhead to improve the generation speed in memory-bound environment. VocabTrim reconstructs the drafter LM head to contain only a limited set of tokens, selected by the most frequently sampled from the vocabulary of the target model. While limiting the vocabulary in drafting slightly degrades the acceptance rate, it significantly reduces the drafting latency in memory-bound process which is often the case on edge devices, resulting in higher memory-bound speed up (MBSU). We show that our method can boost the memory-bound speed-up for Llama-3 models on Spec-Bench, specifically by 16% for Llama-3.2-3B-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03211v1">DistZO2: High-Throughput and Memory-Efficient Zeroth-Order Fine-tuning LLMs with Distributed Parallel Computing</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) remains resource-intensive due to their sheer scale. While zeroth-order (ZO) optimization provides a memory-efficient alternative by eliminating backward passes, its application to multi-hundred-billion-parameter models is constrained by GPU memory and compute throughput. The ZO2 framework addresses the memory bottleneck by offloading model parameters to CPU memory and overlapping transformer block transfer with dual forward computation on a single GPU. However, ZO2 remains limited by its single-device execution and achieves modest throughput. In this work, we present DistZO2, a high-throughput, memory-efficient framework for distributed zeroth-order fine-tuning of LLMs. DistZO2 introduces three parallel strategies: (1) Perturbation Parallelism (PertP), which parallelizes the two perturbed forward passes across devices; (2) Distributed Data Parallelism (DDP), adapted to the scalar-gradient nature of ZO training; and (3) a unified 2D Parallelism design that combines PertP and DDP. To further mitigate communication bottlenecks introduced by parameter offloading, we propose a hardware-aware communication strategy that slices parameter blocks and redistributes them across GPUs via high-speed interconnects such as NVLink. DistZO2 scales zeroth-order fine-tuning to modern multi-GPU systems, preserving ZO2's memory efficiency while substantially improving training throughput. In our experiments on OPT-175B, DistZO2 achieves a 3x speedup over ZO2 with distributed computing. DistZO2's code has been open-sourced in https://github.com/liangyuwang/zo2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03194v1">How Much Content Do LLMs Generate That Induces Cognitive Bias in Users?</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 17 pages, 2 figures. to be submitted to AACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into applications ranging from review summarization to medical diagnosis support, where they affect human decisions. Even though LLMs perform well in many tasks, they may also inherit societal or cognitive biases, which can inadvertently transfer to humans. We investigate when and how LLMs expose users to biased content and quantify its severity. Specifically, we assess three LLM families in summarization and news fact-checking tasks, evaluating how much LLMs stay consistent with their context and/or hallucinate. Our findings show that LLMs expose users to content that changes the sentiment of the context in 21.86% of the cases, hallucinates on post-knowledge-cutoff data questions in 57.33% of the cases, and primacy bias in 5.94% of the cases. We evaluate 18 distinct mitigation methods across three LLM families and find that targeted interventions can be effective. Given the prevalent use of LLMs in high-stakes domains, such as healthcare or legal analysis, our results highlight the need for robust technical safeguards and for developing user-centered interventions that address LLM limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18382v2">One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 31 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Pre-trained Large Language Models (LLMs) have shown promise in solving planning problems but often struggle to ensure plan correctness, especially for long-horizon tasks. Meanwhile, traditional robotic task and motion planning (TAMP) frameworks address these challenges more reliably by combining high-level symbolic search with low-level motion planning. At the core of TAMP is the planning domain, an abstract world representation defined through symbolic predicates and actions. However, creating these domains typically involves substantial manual effort and domain expertise, limiting generalizability. We introduce Planning Domain Derivation with LLMs (PDDLLM), a novel approach that combines simulated physical interaction with LLM reasoning to improve planning performance. The method reduces reliance on humans by inferring planning domains from a single annotated task-execution demonstration. Unlike prior domain-inference methods that rely on partially predefined or language descriptions of planning domains, PDDLLM constructs domains entirely from scratch and automatically integrates them with low-level motion planning skills, enabling fully automated long-horizon planning. PDDLLM is evaluated on over 1,200 diverse tasks spanning nine environments and benchmarked against six LLM-based planning baselines, demonstrating superior long-horizon planning performance, lower token costs, and successful deployment on multiple physical robot platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03162v1">MateInfoUB: A Real-World Benchmark for Testing LLMs in Competitive, Multilingual, and Multimodal Educational Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 14 pages (9 paper, 2 references, 3 annexes). Accepted for BEA 2025!
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has transformed various domains, particularly computer science (CS) education. These models exhibit remarkable capabilities in code-related tasks and problem-solving, raising questions about their potential and limitations in advanced CS contexts. This study presents a novel bilingual (English-Romanian) multimodal (text and image) dataset of multiple-choice questions derived from a high-level computer science competition. A particularity of our dataset is that the problems are conceived such that some of them are easier solved using reasoning on paper, while for others writing code is more efficient. We systematically evaluate State of The Art LLMs on this dataset, analyzing their performance on theoretical programming tasks. Our findings reveal the strengths and limitations of current LLMs, including the influence of language choice (English vs. Romanian), providing insights into their applicability in CS education and competition settings. We also address critical ethical considerations surrounding educational integrity and the fairness of assessments in the context of LLM usage. These discussions aim to inform future educational practices and policies. To support further research, our dataset will be made publicly available in both English and Romanian. Additionally, we release an educational application tailored for Romanian students, enabling them to self-assess using the dataset in an interactive and practice-oriented environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03156v1">The Impact of LLM-Assistants on Software Developer Productivity: A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 37 pages
    </div>
    <details class="paper-abstract">
      Large language model assistants (LLM-assistants) present new opportunities to transform software development. Developers are increasingly adopting these tools across tasks, including coding, testing, debugging, documentation, and design. Yet, despite growing interest, there is no synthesis of how LLM-assistants affect software developer productivity. In this paper, we present a systematic literature review of 37 peer-reviewed studies published between January 2014 and December 2024 that examine this impact. Our analysis reveals that LLM-assistants offer both considerable benefits and critical risks. Commonly reported gains include minimized code search, accelerated development, and the automation of trivial and repetitive tasks. However, studies also highlight concerns around cognitive offloading, reduced team collaboration, and inconsistent effects on code quality. While the majority of studies (92%) adopt a multi-dimensional perspective by examining at least two SPACE dimensions, reflecting increased awareness of the complexity of developer productivity, only 14% extend beyond three dimensions, indicating substantial room for more integrated evaluations. Satisfaction, Performance, and Efficiency are the most frequently investigated dimensions, whereas Communication and Activity remain underexplored. Most studies are exploratory (64%) and methodologically diverse, but lack longitudinal and team-based evaluations. This review surfaces key research gaps and provides recommendations for future research and practice. All artifacts associated with this study are publicly available at https://zenodo.org/records/15788502.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03153v1">HGCA: Hybrid GPU-CPU Attention for Long Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Scaling inference for large language models (LLMs) is increasingly constrained by limited GPU memory, especially due to growing key-value (KV) caches required for long-context generation. While existing approaches offload KV caches to CPU memory or apply sparse attention to reduce GPU load, they often underutilize CPU compute resources and compromise accuracy. We present HGCA, a hybrid CPU-GPU attention mechanism that enables scalable, high-throughput LLM inference with near-full attention quality. HGCA performs dense attention on recently generated KV entries retained in GPU memory and parallel sparse attention on selected, salient KV entries in CPU memory. The attention outputs are efficiently merged using log-sum-exp fusion, minimizing PCIe transfer overhead. HGCA also introduces a finegrained, per-head sparsification strategy optimized for CPU execution, preserving contextual relevance while reducing computation. Our implementation seamlessly integrates into existing LLM frameworks without requiring model retraining. Experiments across diverse models and workloads show that HGCA achieves superior scalability, supports longer sequences and larger batch sizes, and outperforms existing sparse attention baselines in both performance and accuracy -- all on commodity GPU hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05232v3">LIAR: Leveraging Inference Time Alignment (Best-of-N) to Jailbreak LLMs in Seconds</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Jailbreak attacks expose vulnerabilities in safety-aligned LLMs by eliciting harmful outputs through carefully crafted prompts. Existing methods rely on discrete optimization or trained adversarial generators, but are slow, compute-intensive, and often impractical. We argue that these inefficiencies stem from a mischaracterization of the problem. Instead, we frame jailbreaks as inference-time misalignment and introduce LIAR (Leveraging Inference-time misAlignment to jailbReak), a fast, black-box, best-of-$N$ sampling attack requiring no training. LIAR matches state-of-the-art success rates while reducing perplexity by $10\times$ and Time-to-Attack from hours to seconds. We also introduce a theoretical "safety net against jailbreaks" metric to quantify safety alignment strength and derive suboptimality bounds. Our work offers a simple yet effective tool for evaluating LLM robustness and advancing alignment research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03064v1">LLM-Driven Auto Configuration for Transient IoT Device Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Today's Internet of Things (IoT) has evolved from simple sensing and actuation devices to those with embedded processing and intelligent services, enabling rich collaborations between users and their devices. However, enabling such collaboration becomes challenging when transient devices need to interact with host devices in temporarily visited environments. In such cases, fine-grained access control policies are necessary to ensure secure interactions; however, manually implementing them is often impractical for non-expert users. Moreover, at run-time, the system must automatically configure the devices and enforce such fine-grained access control rules. Additionally, the system must address the heterogeneity of devices. In this paper, we present CollabIoT, a system that enables secure and seamless device collaboration in transient IoT environments. CollabIoT employs a Large language Model (LLM)-driven approach to convert users' high-level intents to fine-grained access control policies. To support secure and seamless device collaboration, CollabIoT adopts capability-based access control for authorization and uses lightweight proxies for policy enforcement, providing hardware-independent abstractions. We implement a prototype of CollabIoT's policy generation and auto configuration pipelines and evaluate its efficacy on an IoT testbed and in large-scale emulated environments. We show that our LLM-based policy generation pipeline is able to generate functional and correct policies with 100% accuracy. At runtime, our evaluation shows that our system configures new devices in ~150 ms, and our proxy-based data plane incurs network overheads of up to 2 ms and access control overheads up to 0.3 ms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03052v1">From 2:4 to 8:16 sparsity patterns in LLMs for Outliers and Weights with Variance Correction</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in size, efficient compression techniques like quantization and sparsification are critical. While quantization maintains performance with reduced precision, structured sparsity methods, such as N:M sparsification, often fall short due to limited flexibility, and sensitivity to outlier weights. We explore 8:16 semi-structured sparsity, demonstrating its ability to surpass the Performance Threshold-where a compressed model matches the accuracy of its uncompressed or smaller counterpart under equivalent memory constraints. Compared to 2:4 sparsity, 8:16 offers greater flexibility with minimal storage overhead (0.875 vs. 0.75 bits/element). We also apply sparse structured patterns for salient weights, showing that structured sparsity for outliers is competitive with unstructured approaches leading to equivalent or better results. Finally, we demonstrate that simple techniques such as variance correction and SmoothQuant like weight equalization improve sparse models performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03051v1">Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Improving and understanding the training dynamics and reasoning of Large Language Models (LLMs) has become essential for their deployment in AI-based security tools, such as software vulnerability detection. In this work, we present an extensive study aimed at advancing recent RL-based finetuning techniques for LLMs in the context of vulnerability detection. We start by highlighting key limitations of commonly adopted LLMs, such as their tendency to over-predict certain types of vulnerabilities while failing to detect others. To address this challenge, we explore the use of Group Relative Policy Optimization (GRPO), a recent policy-gradient method, for guiding LLM behavior through structured, rule-based rewards. We enable its application to the vulnerability detection task by redefining its advantage functions and reward signals using annotations from widely used datasets in the field, including BigVul, DiverseVul, and CleanVul. The proposed methodology enables an extensive set of experiments, addressing multiple research questions regarding the impact of GRPO on generalization, reasoning capabilities, and performance improvements over standard supervised finetuning (SFT). Our findings offer valuable insights into the potential of RL-based training to enhance both the performance and reasoning abilities of LLMs in the context of software vulnerability detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03042v1">Dynamic Long Short-Term Memory Based Memory Storage For Long Horizon LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 7 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Memory storage for Large Language models (LLMs) is becoming an increasingly active area of research, particularly for enabling personalization across long conversations. We propose Pref-LSTM, a dynamic and lightweight framework that combines a BERT-based classifier with a LSTM memory module that generates memory embedding which then is soft-prompt injected into a frozen LLM. We synthetically curate a dataset of preference and non-preference conversation turns to train our BERT-based classifier. Although our LSTM-based memory encoder did not yield strong results, we find that the BERT-based classifier performs reliably in identifying explicit and implicit user preferences. Our research demonstrates the viability of using preference filtering with LSTM gating principals as an efficient path towards scalable user preference modeling, without extensive overhead and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02851v1">MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at https://github.com/purbeshmitra/MOTIF and https://huggingface.co/purbeshmitra/MOTIF, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02778v1">Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 31 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have become transformative, they still make mistakes and can explore unproductive reasoning paths. Self-correction is an important capability for a trustworthy LLM, particularly an autoregressive LLM. While LLMs can identify error in user input, they exhibit a systematic 'Self-Correction Blind Spot' - failing to correct identical error in their own outputs. To systematically study this phenomenon, we introduce Self-Correction Bench, a systematic framework to measure this phenomenon through controlled error injection at three complexity levels. Testing 14 models, we find an average 64.5% blind spot rate. We find multiple evidences that this limitation relates to training data composition: human training demonstrations predominantly show error-free responses rather than error-correction sequences, unlike RL-trained models that learn error correction through outcome feedback. Remarkably, simply appending "Wait" reduces blind spots by 89.3%, suggesting that the capability exists but requires activation. Our work highlights a critical limitation in current LLMs and offers potential avenues for improving their reliability and trustworthiness.
    </details>
</div>
