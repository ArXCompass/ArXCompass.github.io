# llm - 2024_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03160v2">User-in-the-loop Evaluation of Multimodal LLMs for Activity Assistance</a></div>
    <div class="paper-meta">
      📅 2024-08-12
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Our research investigates the capability of modern multimodal reasoning models, powered by Large Language Models (LLMs), to facilitate vision-powered assistants for multi-step daily activities. Such assistants must be able to 1) encode relevant visual history from the assistant's sensors, e.g., camera, 2) forecast future actions for accomplishing the activity, and 3) replan based on the user in the loop. To evaluate the first two capabilities, grounding visual history and forecasting in short and long horizons, we conduct benchmarking of two prominent classes of multimodal LLM approaches -- Socratic Models and Vision Conditioned Language Models (VCLMs) on video-based action anticipation tasks using offline datasets. These offline benchmarks, however, do not allow us to close the loop with the user, which is essential to evaluate the replanning capabilities and measure successful activity completion in assistive scenarios. To that end, we conduct a first-of-its-kind user study, with 18 participants performing 3 different multi-step cooking activities while wearing an egocentric observation device called Aria and following assistance from multimodal LLMs. We find that the Socratic approach outperforms VCLMs in both offline and online settings. We further highlight how grounding long visual history, common in activity assistance, remains challenging in current models, especially for VCLMs, and demonstrate that offline metrics do not indicate online performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17728v2">Facilitating Holistic Evaluations with LLMs: Insights from Scenario-Based Experiments</a></div>
    <div class="paper-meta">
      📅 2024-08-12
      | 💬 The final version appears in the proceedings of the 32nd International Conference on Computers in Education (ICCE 2024)
    </div>
    <details class="paper-abstract">
      Workshop courses designed to foster creativity are gaining popularity. However, even experienced faculty teams find it challenging to realize a holistic evaluation that accommodates diverse perspectives. Adequate deliberation is essential to integrate varied assessments, but faculty often lack the time for such exchanges. Deriving an average score without discussion undermines the purpose of a holistic evaluation. Therefore, this paper explores the use of a Large Language Model (LLM) as a facilitator to integrate diverse faculty assessments. Scenario-based experiments were conducted to determine if the LLM could integrate diverse evaluations and explain the underlying pedagogical theories to faculty. The results were noteworthy, showing that the LLM can effectively facilitate faculty discussions. Additionally, the LLM demonstrated the capability to create evaluation criteria by generalizing a single scenario-based experiment, leveraging its already acquired pedagogical domain knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05882v1">Creating Arabic LLM Prompts at Scale</a></div>
    <div class="paper-meta">
      📅 2024-08-12
    </div>
    <details class="paper-abstract">
      The debut of chatGPT and BARD has popularized instruction following text generation using LLMs, where a user can interrogate an LLM using natural language requests and obtain natural language answers that matches their requests. Training LLMs to respond in this manner requires a large number of worked out examples of user requests (aka prompts) with corresponding gold responses. In this paper, we introduce two methods for creating such prompts for Arabic cheaply and quickly. The first methods entails automatically translating existing prompt datasets from English, such as PromptSource and Super-NaturalInstructions, and then using machine translation quality estimation to retain high quality translations only. The second method involves creating natural language prompts on top of existing Arabic NLP datasets. Using these two methods we were able to create more than 67.4 million Arabic prompts that cover a variety of tasks including summarization, headline generation, grammar checking, open/closed question answering, creative writing, etc. We show that fine tuning an open 7 billion parameter large language model, namely base Qwen2 7B, enables it to outperform a state-of-the-art 70 billion parameter instruction tuned model, namely Llama3 70B, in handling Arabic prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05221v2">LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-08-11
      | 💬 Project website: https://www.llm-reasoners.net/
    </div>
    <details class="paper-abstract">
      Generating accurate step-by-step reasoning is essential for Large Language Models (LLMs) to address complex problems and enhance robustness and interpretability. Despite the flux of research on developing advanced reasoning approaches, systematically analyzing the diverse LLMs and reasoning strategies in generating reasoning chains remains a significant challenge. The difficulties stem from the lack of two key elements: (1) an automatic method for evaluating the generated reasoning chains on different tasks, and (2) a unified formalism and implementation of the diverse reasoning approaches for systematic comparison. This paper aims to close the gap: (1) We introduce AutoRace for fully automated reasoning chain evaluation. Existing metrics rely on expensive human annotations or pre-defined LLM prompts not adaptable to different tasks. In contrast, AutoRace automatically creates detailed evaluation criteria tailored for each task, and uses GPT-4 for accurate evaluation following the criteria. (2) We develop LLM Reasoners, a library for standardized modular implementation of existing and new reasoning algorithms, under a unified formulation of the search, reward, and world model components. With the new evaluation and library, (3) we conduct extensive study of different reasoning approaches (e.g., CoT, ToT, RAP). The analysis reveals interesting findings about different factors contributing to reasoning, including the reward-guidance, breadth-vs-depth in search, world model, and prompt formats, etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08899v1">Kov: Transferable and Naturalistic Black-Box LLM Attacks using Markov Decision Processes and Tree Search</a></div>
    <div class="paper-meta">
      📅 2024-08-11
    </div>
    <details class="paper-abstract">
      Eliciting harmful behavior from large language models (LLMs) is an important task to ensure the proper alignment and safety of the models. Often when training LLMs, ethical guidelines are followed yet alignment failures may still be uncovered through red teaming adversarial attacks. This work frames the red-teaming problem as a Markov decision process (MDP) and uses Monte Carlo tree search to find harmful behaviors of black-box, closed-source LLMs. We optimize token-level prompt suffixes towards targeted harmful behaviors on white-box LLMs and include a naturalistic loss term, log-perplexity, to generate more natural language attacks for better interpretability. The proposed algorithm, Kov, trains on white-box LLMs to optimize the adversarial attacks and periodically evaluates responses from the black-box LLM to guide the search towards more harmful black-box behaviors. In our preliminary study, results indicate that we can jailbreak black-box models, such as GPT-3.5, in only 10 queries, yet fail on GPT-4$-$which may indicate that newer models are more robust to token-level attacks. All work to reproduce these results is open sourced (https://github.com/sisl/Kov.jl).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01365v3">Prompt-prompted Adaptive Structured Pruning for Efficient LLM Generation</a></div>
    <div class="paper-meta">
      📅 2024-08-11
      | 💬 Revision 1: Updated abstract with code link; re-ran top-k + sampling rows in Table 4, conclusions unchanged Revision 2: Reframing and new experiments, conclusions unchanged
    </div>
    <details class="paper-abstract">
      With the development of transformer-based large language models (LLMs), they have been applied to many fields due to their remarkable utility, but this comes at a considerable computational cost at deployment. Fortunately, some methods such as pruning or constructing a mixture of experts (MoE) aim at exploiting sparsity in transformer feedforward (FF) blocks to gain boosts in speed and reduction in memory requirements. However, these techniques can be very costly and inflexible in practice, as they often require training or are restricted to specific types of architectures. To address this, we introduce GRIFFIN, a novel training-free and calibration-free method that selects unique FF experts at the sequence level for efficient generation across a plethora of LLMs with different non-ReLU activation functions. This is possible due to a critical observation that many trained LLMs naturally produce highly structured FF activation patterns within a sequence, which we call flocking. Despite our method's simplicity, we show with 50% of the FF parameters, GRIFFIN maintains the original model's performance with little to no degradation on a variety of classification and generation tasks, all while improving latency (e.g. 1.29$\times$ and 1.25$\times$ speed-ups in Gemma 7B and Llama 2 13B, respectively, on an NVIDIA L40). Code is available at https://github.com/hdong920/GRIFFIN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09613v3">Digital Socrates: Evaluating LLMs through Explanation Critiques</a></div>
    <div class="paper-meta">
      📅 2024-08-11
      | 💬 ACL 2024
    </div>
    <details class="paper-abstract">
      While LLMs can provide reasoned explanations along with their answers, the nature and quality of those explanations are still poorly understood. In response, our goal is to define a detailed way of characterizing the explanation capabilities of modern models and to create a nuanced, interpretable explanation evaluation tool that can generate such characterizations automatically, without relying on expensive API calls or human annotations. Our approach is to (a) define the new task of explanation critiquing - identifying and categorizing any main flaw in an explanation and providing suggestions to address the flaw, (b) create a sizeable, human-verified dataset for this task, and (c) train an open-source, automatic critique model (called Digital Socrates) using this data. Through quantitative and qualitative analysis, we demonstrate how Digital Socrates is useful for revealing insights about student models by examining their reasoning chains, and how it can provide high-quality, nuanced, automatic evaluation of those model explanations for the first time. Digital Socrates thus fills an important gap in evaluation tools for understanding and improving the explanation behavior of models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05676v1">A Decoding Acceleration Framework for Industrial Deployable LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2024-08-11
    </div>
    <details class="paper-abstract">
      Recently, increasing attention has been paid to LLM-based recommender systems, but their deployment is still under exploration in the industry. Most deployments utilize LLMs as feature enhancers, generating augmentation knowledge in the offline stage. However, in recommendation scenarios, involving numerous users and items, even offline generation with LLMs consumes considerable time and resources. This generation inefficiency stems from the autoregressive nature of LLMs, and a promising direction for acceleration is speculative decoding, a Draft-then-Verify paradigm that increases the number of generated tokens per decoding step. In this paper, we first identify that recommendation knowledge generation is suitable for retrieval-based speculative decoding. Then, we discern two characteristics: (1) extensive items and users in RSs bring retrieval inefficiency, and (2) RSs exhibit high diversity tolerance for text generated by LLMs. Based on the above insights, we propose a Decoding Acceleration Framework for LLM-based Recommendation (dubbed DARE), with Customized Retrieval Pool to improve retrieval efficiency and Relaxed Verification to increase the acceptance rate of draft tokens, respectively. Extensive experiments demonstrate that DARE achieves a 3-5x speedup and is compatible with various frameworks and backbone LLMs. DARE has also been deployed to online advertising scenarios within a large-scale commercial environment, achieving a 3.45x speedup while maintaining the downstream performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05534v1">Can LLMs Replace Manual Annotation of Software Engineering Artifacts?</a></div>
    <div class="paper-meta">
      📅 2024-08-10
    </div>
    <details class="paper-abstract">
      Experimental evaluations of software engineering innovations, e.g., tools and processes, often include human-subject studies as a component of a multi-pronged strategy to obtain greater generalizability of the findings. However, human-subject studies in our field are challenging, due to the cost and difficulty of finding and employing suitable subjects, ideally, professional programmers with varying degrees of experience. Meanwhile, large language models (LLMs) have recently started to demonstrate human-level performance in several areas. This paper explores the possibility of substituting costly human subjects with much cheaper LLM queries in evaluations of code and code-related artifacts. We study this idea by applying six state-of-the-art LLMs to ten annotation tasks from five datasets created by prior work, such as judging the accuracy of a natural language summary of a method or deciding whether a code change fixes a static analysis warning. Our results show that replacing some human annotation effort with LLMs can produce inter-rater agreements equal or close to human-rater agreement. To help decide when and how to use LLMs in human-subject studies, we propose model-model agreement as a predictor of whether a given task is suitable for LLMs at all, and model confidence as a means to select specific samples where LLMs can safely replace human annotators. Overall, our work is the first step toward mixed human-LLM evaluations in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05499v1">LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale</a></div>
    <div class="paper-meta">
      📅 2024-08-10
      | 💬 15 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Recently, there has been an extensive research effort in building efficient large language model (LLM) inference serving systems. These efforts not only include innovations in the algorithm and software domains but also constitute developments of various hardware acceleration techniques. Nevertheless, there is a lack of simulation infrastructure capable of accurately modeling versatile hardware-software behaviors in LLM serving systems without extensively extending the simulation time. This paper aims to develop an effective simulation tool, called LLMServingSim, to support future research in LLM serving systems. In designing LLMServingSim, we focus on two limitations of existing simulators: (1) they lack consideration of the dynamic workload variations of LLM inference serving due to its autoregressive nature, and (2) they incur repetitive simulations without leveraging algorithmic redundancies in LLMs. To address these limitations, LLMServingSim simulates the LLM serving in the granularity of iterations, leveraging the computation redundancies across decoder blocks and reusing the simulation results from previous iterations. Additionally, LLMServingSim provides a flexible framework that allows users to plug in any accelerator compiler-and-simulation stacks for exploring various system designs with heterogeneous processors. Our experiments demonstrate that LLMServingSim produces simulation results closely following the performance behaviors of real GPU-based LLM serving system with less than 14.7% error rate, while offering 91.5x faster simulation speed compared to existing accelerator simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05456v1">Path-LLM: A Shortest-Path-based LLM Learning for Unified Graph Representation</a></div>
    <div class="paper-meta">
      📅 2024-08-10
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Unified graph representation learning aims to produce node embeddings, which can be applied to multiple downstream applications. However, existing studies based on graph neural networks and language models either suffer from the limitations of numerous training needed toward specific downstream predictions or have shallow semantic features. In this work, we propose a novel Path-LLM model to learn unified graph representation, which leverages a powerful large language model (LLM) to incorporate our proposed path features. Our Path-LLM framework consists of several well-designed techniques. First, we develop a new mechanism of long-to-short shortest path (L2SP) selection, which covers essential connections between different dense groups. An in-depth comparison of different path selection plans is offered to illustrate the strength of our designed L2SP. Then, we design path textualization to obtain L2SP-based training texts. Next, we feed the texts into a self-supervised LLM training process to learn embeddings. Extensive experiments on benchmarks validate the superiority of Path-LLM against the state-of-the-art WalkLM method on two classical graph learning tasks (node classification and link prediction) and one NP-hard graph query processing task (keyword search), meanwhile saving more than 90% of training paths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08896v1">LLMJudge: LLMs for Relevance Judgments</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 LLMJudge Challenge Overview, 3 pages
    </div>
    <details class="paper-abstract">
      The LLMJudge challenge is organized as part of the LLM4Eval workshop at SIGIR 2024. Test collections are essential for evaluating information retrieval (IR) systems. The evaluation and tuning of a search system is largely based on relevance labels, which indicate whether a document is useful for a specific search and user. However, collecting relevance judgments on a large scale is costly and resource-intensive. Consequently, typical experiments rely on third-party labelers who may not always produce accurate annotations. The LLMJudge challenge aims to explore an alternative approach by using LLMs to generate relevance judgments. Recent studies have shown that LLMs can generate reliable relevance judgments for search systems. However, it remains unclear which LLMs can match the accuracy of human labelers, which prompts are most effective, how fine-tuned open-source LLMs compare to closed-source LLMs like GPT-4, whether there are biases in synthetically generated data, and if data leakage affects the quality of generated labels. This challenge will investigate these questions, and the collected data will be released as a package to support automatic relevance judgment research in information retrieval and search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09805v3">DocMath-Eval: Evaluating Math Reasoning Capabilities of LLMs in Understanding Long and Specialized Documents</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 ACL 2024 Oral
    </div>
    <details class="paper-abstract">
      Recent LLMs have demonstrated remarkable performance in solving exam-like math word problems. However, the degree to which these numerical reasoning skills are effective in real-world scenarios, particularly in expert domains, is still largely unexplored. This paper introduces DocMath-Eval, a comprehensive benchmark specifically designed to evaluate the numerical reasoning capabilities of LLMs in the context of understanding and analyzing specialized documents containing both text and tables. We conduct an extensive evaluation of 48 LLMs with Chain-of-Thought and Program-of-Thought prompting methods, aiming to comprehensively assess the capabilities and limitations of existing LLMs in DocMath-Eval. We found that even the current best-performing system (i.e., GPT-4o) still significantly lags behind human experts in solving complex numerical reasoning problems grounded in long contexts. We believe that DocMath-Eval can serve as a valuable benchmark for evaluating LLMs' capabilities in solving challenging numerical reasoning problems within expert domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18783v3">Psychological Profiling in Cybersecurity: A Look at LLMs and Psycholinguistic Features</a></div>
    <div class="paper-meta">
      📅 2024-08-09
    </div>
    <details class="paper-abstract">
      The increasing sophistication of cyber threats necessitates innovative approaches to cybersecurity. In this paper, we explore the potential of psychological profiling techniques, particularly focusing on the utilization of Large Language Models (LLMs) and psycholinguistic features. We investigate the intersection of psychology and cybersecurity, discussing how LLMs can be employed to analyze textual data for identifying psychological traits of threat actors. We explore the incorporation of psycholinguistic features, such as linguistic patterns and emotional cues, into cybersecurity frameworks. Our research underscores the importance of integrating psychological perspectives into cybersecurity practices to bolster defense mechanisms against evolving threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07311v8">Knowledge Graph Large Language Model (KG-LLM) for Link Prediction</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The task of multi-hop link prediction within knowledge graphs (KGs) stands as a challenge in the field of knowledge graph analysis, as it requires the model to reason through and understand all intermediate connections before making a prediction. In this paper, we introduce the Knowledge Graph Large Language Model (KG-LLM), a novel framework that leverages large language models (LLMs) for knowledge graph tasks. We first convert structured knowledge graph data into natural language and then use these natural language prompts to fine-tune LLMs to enhance multi-hop link prediction in KGs. By converting the KG to natural language prompts, our framework is designed to learn the latent representations of entities and their interrelations. To show the efficacy of the KG-LLM Framework, we fine-tune three leading LLMs within this framework, including Flan-T5, LLaMa2 and Gemma. Further, we explore the framework's potential to provide LLMs with zero-shot capabilities for handling previously unseen prompts. Experimental results show that KG-LLM significantly improves the models' generalization capabilities, leading to more accurate predictions in unfamiliar scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05102v1">How Well Do LLMs Identify Cultural Unity in Diversity?</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 COLM 2024
    </div>
    <details class="paper-abstract">
      Much work on the cultural awareness of large language models (LLMs) focuses on the models' sensitivity to geo-cultural diversity. However, in addition to cross-cultural differences, there also exists common ground across cultures. For instance, a bridal veil in the United States plays a similar cultural-relevant role as a honggaitou in China. In this study, we introduce a benchmark dataset CUNIT for evaluating decoder-only LLMs in understanding the cultural unity of concepts. Specifically, CUNIT consists of 1,425 evaluation examples building upon 285 traditional cultural-specific concepts across 10 countries. Based on a systematic manual annotation of cultural-relevant features per concept, we calculate the cultural association between any pair of cross-cultural concepts. Built upon this dataset, we design a contrastive matching task to evaluate the LLMs' capability to identify highly associated cross-cultural concept pairs. We evaluate 3 strong LLMs, using 3 popular prompting strategies, under the settings of either giving all extracted concept features or no features at all on CUNIT Interestingly, we find that cultural associations across countries regarding clothing concepts largely differ from food. Our analysis shows that LLMs are still limited to capturing cross-cultural associations between concepts compared to humans. Moreover, geo-cultural proximity shows a weak influence on model performance in capturing cross-cultural associations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05101v1">MooER: LLM-based Speech Recognition and Translation Models from Moore Threads</a></div>
    <div class="paper-meta">
      📅 2024-08-09
    </div>
    <details class="paper-abstract">
      In this paper, we present MooER, a LLM-based large-scale automatic speech recognition (ASR) / automatic speech translation (AST) model of Moore Threads. A 5000h pseudo labeled dataset containing open source and self collected speech data is used for training. We achieve performance comparable to other open source models trained with up to hundreds of thousands of hours of labeled speech data. Meanwhile, experiments conducted on Covost2 Zh2en testset suggest that our model outperforms other open source Speech LLMs. A BLEU score of 25.2 can be obtained. The main contributions of this paper are summarized as follows. First, this paper presents a training strategy for encoders and LLMs on speech related tasks (including ASR and AST) using a small size of pseudo labeled data without any extra manual annotation and selection. Second, we release our ASR and AST models and plan to open-source our training code and strategy in the near future. Moreover, a model trained on 8wh scale training data is planned to be released later on.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05035v1">Examining the Behavior of LLM Architectures Within the Framework of Standardized National Exams in Brazil</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 Accepted at the Seventh AAAI/ACM Conference on AI, Ethics and Society (AIES 2024). 14 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The Exame Nacional do Ensino M\'edio (ENEM) is a pivotal test for Brazilian students, required for admission to a significant number of universities in Brazil. The test consists of four objective high-school level tests on Math, Humanities, Natural Sciences and Languages, and one writing essay. Students' answers to the test and to the accompanying socioeconomic status questionnaire are made public every year (albeit anonymized) due to transparency policies from the Brazilian Government. In the context of large language models (LLMs), these data lend themselves nicely to comparing different groups of humans with AI, as we can have access to human and machine answer distributions. We leverage these characteristics of the ENEM dataset and compare GPT-3.5 and 4, and MariTalk, a model trained using Portuguese data, to humans, aiming to ascertain how their answers relate to real societal groups and what that may reveal about the model biases. We divide the human groups by using socioeconomic status (SES), and compare their answer distribution with LLMs for each question and for the essay. We find no significant biases when comparing LLM performance to humans on the multiple-choice Brazilian Portuguese tests, as the distance between model and human answers is mostly determined by the human accuracy. A similar conclusion is found by looking at the generated text as, when analyzing the essays, we observe that human and LLM essays differ in a few key factors, one being the choice of words where model essays were easily separable from human ones. The texts also differ syntactically, with LLM generated essays exhibiting, on average, smaller sentences and less thought units, among other differences. These results suggest that, for Brazilian Portuguese in the ENEM context, LLM outputs represent no group of humans, being significantly different from the answers from Brazilian students across all tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05019v1">Instruction Tuning-free Visual Token Complement for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 Accepted by ECCV2024 (20pages)
    </div>
    <details class="paper-abstract">
      As the open community of large language models (LLMs) matures, multimodal LLMs (MLLMs) have promised an elegant bridge between vision and language. However, current research is inherently constrained by challenges such as the need for high-quality instruction pairs and the loss of visual information in image-to-text training objectives. To this end, we propose a Visual Token Complement framework (VTC) that helps MLLMs regain the missing visual features and thus improve response accuracy. Specifically, our VTC integrates text-to-image generation as a guide to identifying the text-irrelevant features, and a visual selector is then developed to generate complementary visual tokens to enrich the original visual input. Moreover, an iterative strategy is further designed to extract more visual information by iteratively using the visual selector without any additional training. Notably, the training pipeline requires no additional image-text pairs, resulting in a desired instruction tuning-free property. Both qualitative and quantitative experiments demonstrate the superiority and efficiency of our VTC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21424v2">Cost-Effective Hallucination Detection for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 Accepted to GenAI Evaluation Workshop at KDD 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be prone to hallucinations - generating unreliable outputs that are unfaithful to their inputs, external facts or internally inconsistent. In this work, we address several challenges for post-hoc hallucination detection in production settings. Our pipeline for hallucination detection entails: first, producing a confidence score representing the likelihood that a generated answer is a hallucination; second, calibrating the score conditional on attributes of the inputs and candidate response; finally, performing detection by thresholding the calibrated score. We benchmark a variety of state-of-the-art scoring methods on different datasets, encompassing question answering, fact checking, and summarization tasks. We employ diverse LLMs to ensure a comprehensive assessment of performance. We show that calibrating individual scoring methods is critical for ensuring risk-aware downstream decision making. Based on findings that no individual score performs best in all situations, we propose a multi-scoring framework, which combines different scores and achieves top performance across all datasets. We further introduce cost-effective multi-scoring, which can match or even outperform more expensive detection methods, while significantly reducing computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11063v1">Tabular Transfer Learning via Prompting LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 COLM 2024
    </div>
    <details class="paper-abstract">
      Learning with a limited number of labeled data is a central problem in real-world applications of machine learning, as it is often expensive to obtain annotations. To deal with the scarcity of labeled data, transfer learning is a conventional approach; it suggests to learn a transferable knowledge by training a neural network from multiple other sources. In this paper, we investigate transfer learning of tabular tasks, which has been less studied and successful in the literature, compared to other domains, e.g., vision and language. This is because tables are inherently heterogeneous, i.e., they contain different columns and feature spaces, making transfer learning difficult. On the other hand, recent advances in natural language processing suggest that the label scarcity issue can be mitigated by utilizing in-context learning capability of large language models (LLMs). Inspired by this and the fact that LLMs can also process tables within a unified language space, we ask whether LLMs can be effective for tabular transfer learning, in particular, under the scenarios where the source and target datasets are of different format. As a positive answer, we propose a novel tabular transfer learning framework, coined Prompt to Transfer (P2T), that utilizes unlabeled (or heterogeneous) source data with LLMs. Specifically, P2T identifies a column feature in a source dataset that is strongly correlated with a target task feature to create examples relevant to the target task, thus creating pseudo-demonstrations for prompts. Experimental results demonstrate that P2T outperforms previous methods on various tabular learning benchmarks, showing good promise for the important, yet underexplored tabular transfer learning problem. Code is available at https://github.com/jaehyun513/P2T.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19760v2">Legal Minds, Algorithmic Decisions: How LLMs Apply Constitutional Principles in Complex Scenarios</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 Accepted at AIES24
    </div>
    <details class="paper-abstract">
      In this paper, we conduct an empirical analysis of how large language models (LLMs), specifically GPT-4, interpret constitutional principles in complex decision-making scenarios. We examine rulings from the Italian Constitutional Court on bioethics issues that involve trade-offs between competing values and compare model-generated legal arguments on these issues to those presented by the State, the Court, and the applicants. Our results indicate that GPT-4 consistently aligns more closely with progressive interpretations of the Constitution, often overlooking competing values and mirroring the applicants' views rather than the more conservative perspectives of the State or the Court's moderate positions. Our experiments reveal a distinct tendency of GPT-4 to favor progressive legal interpretations, underscoring the influence of underlying data biases. We thus underscore the importance of testing alignment in real-world scenarios and considering the implications of deploying LLMs in decision-making processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12459v2">TrajCogn: Leveraging LLMs for Cognizing Movement Patterns and Travel Purposes from Trajectories</a></div>
    <div class="paper-meta">
      📅 2024-08-09
    </div>
    <details class="paper-abstract">
      Spatio-temporal trajectories are crucial in various data mining tasks. It is important to develop a versatile trajectory learning method that performs different tasks with high accuracy. This involves effectively extracting two core aspects of information--movement patterns and travel purposes--from trajectories. However, this is challenging due to limitations in model capacity and the quality and scale of trajectory datasets. Meanwhile, large language models (LLMs) have shown great success in versatility by training on large-scale, high-quality datasets. Given the similarities between trajectories and sentences, there's potential to leverage LLMs to develop an effective trajectory learning method. However, standard LLMs are not designed to handle the unique spatio-temporal features of trajectories and cannot extract movement patterns and travel purposes. To address these challenges, we propose a model called TrajCogn that effectively utilizes LLMs to model trajectories. TrajCogn leverages the strengths of LLMs to create a versatile trajectory learning approach while addressing the limitations of standard LLMs. First, TrajCogn incorporates a novel trajectory semantic embedder that enables LLMs to process spatio-temporal features and extract movement patterns and travel purposes. Second, TrajCogn introduces a new trajectory prompt that integrates these patterns and purposes into LLMs, allowing the model to adapt to various tasks. Extensive experiments on two real-world datasets and two representative tasks demonstrate that TrajCogn successfully achieves its design goals. Codes are available at https://anonymous.4open.science/r/TrajCogn-5021.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09007v4">LLMind: Orchestrating AI and IoT with LLM for Complex Task Execution</a></div>
    <div class="paper-meta">
      📅 2024-08-09
    </div>
    <details class="paper-abstract">
      Task-oriented communications are an important element in future intelligent IoT systems. Existing IoT systems, however, are limited in their capacity to handle complex tasks, particularly in their interactions with humans to accomplish these tasks. In this paper, we present LLMind, an LLM-based task-oriented AI agent framework that enables effective collaboration among IoT devices, with humans communicating high-level verbal instructions, to perform complex tasks. Inspired by the functional specialization theory of the brain, our framework integrates an LLM with domain-specific AI modules, enhancing its capabilities. Complex tasks, which may involve collaborations of multiple domain-specific AI modules and IoT devices, are executed through a control script generated by the LLM using a Language-Code transformation approach, which first converts language descriptions to an intermediate finite-state machine (FSM) before final precise transformation to code. Furthermore, the framework incorporates a novel experience accumulation mechanism to enhance response speed and effectiveness, allowing the framework to evolve and become progressively sophisticated through continuing user and machine interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04867v1">An Evaluation of Standard Statistical Models and LLMs on Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2024-08-09
    </div>
    <details class="paper-abstract">
      This research examines the use of Large Language Models (LLMs) in predicting time series, with a specific focus on the LLMTIME model. Despite the established effectiveness of LLMs in tasks such as text generation, language translation, and sentiment analysis, this study highlights the key challenges that large language models encounter in the context of time series prediction. We assess the performance of LLMTIME across multiple datasets and introduce classical almost periodic functions as time series to gauge its effectiveness. The empirical results indicate that while large language models can perform well in zero-shot forecasting for certain datasets, their predictive accuracy diminishes notably when confronted with diverse time series data and traditional signals. The primary finding of this study is that the predictive capacity of LLMTIME, similar to other LLMs, significantly deteriorates when dealing with time series data that contain both periodic and trend components, as well as when the signal comprises complex frequency components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00575v2">Instruction-tuning Aligns LLMs to the Human Brain</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 COLM 2024
    </div>
    <details class="paper-abstract">
      Instruction-tuning is a widely adopted finetuning method that enables large language models (LLMs) to generate output that more closely resembles human responses. However, no studies have shown that instruction-tuning actually teaches LLMs to process language in a similar manner as humans. We investigate the effect of instruction-tuning on aligning LLM and human language processing mechanisms in two ways: (1) brain alignment, the similarity of LLM internal representations to neural activity in the human language system, and (2) behavioral alignment, the similarity of LLM and human behavior on a reading task. We assess 25 vanilla and instruction-tuned LLMs on three datasets involving humans reading naturalistic stories and sentences, and find that instruction-tuning generally enhances brain alignment (~6%), but has no similar effect on behavioral alignment. To identify factors underlying this improvement in brain alignment, we compute correlations between brain alignment and various LLM properties, such as model size, problem-solving, and world knowledge understanding. Notably, we find a strong positive correlation between brain alignment and model size (r = 0.95), as well as performance on tasks requiring world knowledge (r = 0.81). Our results demonstrate that instruction-tuning LLMs improves both world knowledge representations and brain alignment, suggesting that the mechanisms that encode world knowledge in LLMs also improve representational alignment to the human brain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17150v2">SimCT: A Simple Consistency Test Protocol in LLMs Development Lifecycle</a></div>
    <div class="paper-meta">
      📅 2024-08-09
    </div>
    <details class="paper-abstract">
      In this work, we report our efforts to advance the standard operation procedure of developing Large Language Models (LLMs) or LLMs-based systems or services in industry. We introduce the concept of Large Language Model Development Lifecycle (LDLC) and then highlight the importance of consistency test in ensuring the delivery quality. The principled solution of consistency test, however, is usually overlooked by industrial practitioners and not urgent in academia, and current practical solutions are insufficiently rigours and labor-intensive. We thus propose a simple yet effective consistency test protocol, named SimCT. SimCT is mainly to proactively check the consistency across different development stages of "bare metal" LLMs or associated services without accessing the model artifacts, in an attempt to expedite the delivery by reducing the back-and-forth alignment communications among multiple teams involved in different development stages. Specifically, SimCT encompasses response-wise and model-wise tests. We implement the protocol with LightGBM and Student's t-test for two components respectively, and perform extensive experiments to substantiate the effectiveness of SimCT and the involved components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13356v2">Large Language Models (LLMs) Assisted Wireless Network Deployment in Urban Settings</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized language understanding and human-like text generation, drawing interest from many other fields with this question in mind: What else are the LLMs capable of? Despite their widespread adoption, ongoing research continues to explore new ways to integrate LLMs into diverse systems. This paper explores new techniques to harness the power of LLMs for 6G (6th Generation) wireless communication technologies, a domain where automation and intelligent systems are pivotal. The inherent adaptability of LLMs to domain-specific tasks positions them as prime candidates for enhancing wireless systems in the 6G landscape. We introduce a novel Reinforcement Learning (RL) based framework that leverages LLMs for network deployment in wireless communications. Our approach involves training an RL agent, utilizing LLMs as its core, in an urban setting to maximize coverage. The agent's objective is to navigate the complexities of urban environments and identify the network parameters for optimal area coverage. Additionally, we integrate LLMs with Convolutional Neural Networks (CNNs) to capitalize on their strengths while mitigating their limitations. The Deep Deterministic Policy Gradient (DDPG) algorithm is employed for training purposes. The results suggest that LLM-assisted models can outperform CNN-based models in some cases while performing at least as well in others.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17649v3">Beyond prompt brittleness: Evaluating the reliability and consistency of political worldviews in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 12 pages, TACL
    </div>
    <details class="paper-abstract">
      Due to the widespread use of large language models (LLMs), we need to understand whether they embed a specific "worldview" and what these views reflect. Recent studies report that, prompted with political questionnaires, LLMs show left-liberal leanings (Feng et al., 2023; Motoki et al., 2024). However, it is as yet unclear whether these leanings are reliable (robust to prompt variations) and whether the leaning is consistent across policies and political leaning. We propose a series of tests which assess the reliability and consistency of LLMs' stances on political statements based on a dataset of voting-advice questionnaires collected from seven EU countries and annotated for policy issues. We study LLMs ranging in size from 7B to 70B parameters and find that their reliability increases with parameter count. Larger models show overall stronger alignment with left-leaning parties but differ among policy programs: They show a (left-wing) positive stance towards environment protection, social welfare state and liberal society but also (right-wing) law and order, with no consistent preferences in the areas of foreign policy and migration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.03720v4">SteP: Stacked LLM Policies for Web Actions</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 Accepted at Conference on Language Modeling (COLM) 2024. 30 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Performing tasks on the web presents fundamental challenges to large language models (LLMs), including combinatorially large open-world tasks and variations across web interfaces. Simply specifying a large prompt to handle all possible behaviors and states is extremely complex, and results in behavior leaks between unrelated behaviors. Decomposition to distinct policies can address this challenge, but requires carefully handing off control between policies. We propose Stacked LLM Policies for Web Actions (SteP), an approach to dynamically compose policies to solve a diverse set of web tasks. SteP defines a Markov Decision Process where the state is a stack of policies representing the control state, i.e., the chain of policy calls. Unlike traditional methods that are restricted to static hierarchies, SteP enables dynamic control that adapts to the complexity of the task. We evaluate SteP against multiple baselines and web environments including WebArena, MiniWoB++, and a CRM. On WebArena, SteP improves (14.9\% to 33.5\%) over SOTA that use GPT-4 policies, while on MiniWob++, SteP is competitive with prior works while using significantly less data. Our code and data are available at https://asappresearch.github.io/webagents-step.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07174v2">LUNAR: Unsupervised LLM-based Log Parsing</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      Log parsing serves as an essential prerequisite for various log analysis tasks. Recent advancements in this field have improved parsing accuracy by leveraging the semantics in logs through fine-tuning large language models (LLMs) or learning from in-context demonstrations. However, these methods heavily depend on labeled examples to achieve optimal performance. In practice, collecting sufficient labeled data is challenging due to the large scale and continuous evolution of logs, leading to performance degradation of existing log parsers after deployment. To address this issue, we propose LUNAR, an unsupervised LLM-based method for efficient and off-the-shelf log parsing. Our key insight is that while LLMs may struggle with direct log parsing, their performance can be significantly enhanced through comparative analysis across multiple logs that differ only in their parameter parts. We refer to such groups of logs as Log Contrastive Units (LCUs). Given the vast volume of logs, obtaining LCUs is difficult. Therefore, LUNAR introduces a hybrid ranking scheme to effectively search for LCUs by jointly considering the commonality and variability among logs. Additionally, LUNAR crafts a novel parsing prompt for LLMs to identify contrastive patterns and extract meaningful log structures from LCUs. Experiments on large-scale public datasets demonstrate that LUNAR significantly outperforms state-of-the-art log parsers in terms of accuracy and efficiency, providing an effective and scalable solution for real-world deployment. \footnote{The code and data are available at \url{https://github.com/Jun-jie-Huang/LUNAR}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04693v1">Understanding the Performance and Estimating the Cost of LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 10 pages, conference
    </div>
    <details class="paper-abstract">
      Due to the cost-prohibitive nature of training Large Language Models (LLMs), fine-tuning has emerged as an attractive alternative for specializing LLMs for specific tasks using limited compute resources in a cost-effective manner. In this paper, we characterize sparse Mixture of Experts (MoE) based LLM fine-tuning to understand their accuracy and runtime performance on a single GPU. Our evaluation provides unique insights into the training efficacy of sparse and dense versions of MoE models, as well as their runtime characteristics, including maximum batch size, execution time breakdown, end-to-end throughput, GPU hardware utilization, and load distribution. Our study identifies the optimization of the MoE layer as crucial for further improving the performance of LLM fine-tuning. Using our profiling results, we also develop and validate an analytical model to estimate the cost of LLM fine-tuning on the cloud. This model, based on parameters of the model and GPU architecture, estimates LLM throughput and the cost of training, aiding practitioners in industry and academia to budget the cost of fine-tuning a specific model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04477v1">What You Need is What You Get: Theory of Mind for an LLM-Based Code Understanding Assistant</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 Accepted at the NIER track of the International Conference on Software Maintenance and Evolution (ICSME), 2024
    </div>
    <details class="paper-abstract">
      A growing number of tools have used Large Language Models (LLMs) to support developers' code understanding. However, developers still face several barriers to using such tools, including challenges in describing their intent in natural language, interpreting the tool outcome, and refining an effective prompt to obtain useful information. In this study, we designed an LLM-based conversational assistant that provides a personalized interaction based on inferred user mental state (e.g., background knowledge and experience). We evaluate the approach in a within-subject study with fourteen novices to capture their perceptions and preferences. Our results provide insights for researchers and tool builders who want to create or improve LLM-based conversational assistants to support novices in code understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04331v1">Enhancing Journalism with AI: A Study of Contextualized Image Captioning for News Articles using LLMs and LMMs</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and large multimodal models (LMMs) have significantly impacted the AI community, industry, and various economic sectors. In journalism, integrating AI poses unique challenges and opportunities, particularly in enhancing the quality and efficiency of news reporting. This study explores how LLMs and LMMs can assist journalistic practice by generating contextualised captions for images accompanying news articles. We conducted experiments using the GoodNews dataset to evaluate the ability of LMMs (BLIP-2, GPT-4v, or LLaVA) to incorporate one of two types of context: entire news articles, or extracted named entities. In addition, we compared their performance to a two-stage pipeline composed of a captioning model (BLIP-2, OFA, or ViT-GPT2) with post-hoc contextualisation with LLMs (GPT-4 or LLaMA). We assess a diversity of models, and we find that while the choice of contextualisation model is a significant factor for the two-stage pipelines, this is not the case in the LMMs, where smaller, open-source models perform well compared to proprietary, GPT-powered ones. Additionally, we found that controlling the amount of provided context enhances performance. These results highlight the limitations of a fully automated approach and underscore the necessity for an interactive, human-in-the-loop strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04323v1">Towards SLO-Optimized LLM Serving via Automatic Inference Engine Tuning</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      A service-level objective (SLO) is a target performance metric of service that cloud vendors aim to ensure. Delivering optimized SLOs can enhance user satisfaction and improve the competitiveness of cloud vendors. As large language models (LLMs) are gaining increasing popularity across various fields, it is of great significance to optimize SLOs for LLM inference services. In this paper, we observe that adjusting the parameters of LLM inference engines can improve service performance, and the optimal parameter configurations of different services are different. Therefore, we propose SCOOT, an automatic performance tuning system to optimize SLOs for each LLM inference service by tuning the parameters of the inference engine. We first propose a generalized formulation of the tuning problem to handle various objectives and constraints between parameters, and SCOOT exploits the Bayesian optimization (BO) technique to resolve the problem via exploration and exploitation. Moreover, SCOOT adopts a random forest to learn hidden constraints during the tuning process to mitigate invalid exploration. To improve the tuning efficiency, SCOOT utilizes the parallel suggestion to accelerate the tuning process. Extensive experiments demonstrate that SCOOT can significantly outperform existing tuning techniques in SLO optimization while greatly improving the tuning efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18349v3">Rejection Improves Reliability: Training LLMs to Refuse Unknown Questions Using RL from Knowledge Feedback</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate erroneous outputs, known as hallucinations, due to their limitations in discerning questions beyond their knowledge scope. While addressing hallucination has been a focal point in research, previous efforts primarily concentrate on enhancing correctness without giving due consideration to the significance of rejection mechanisms. In this paper, we conduct a comprehensive examination of the role of rejection, introducing the notion of model reliability along with corresponding metrics. These metrics measure the model's ability to provide accurate responses while adeptly rejecting questions exceeding its knowledge boundaries, thereby minimizing hallucinations. To improve the inherent reliability of LLMs, we present a novel alignment framework called Reinforcement Learning from Knowledge Feedback (RLKF). RLKF leverages knowledge feedback to dynamically determine the model's knowledge boundary and trains a reliable reward model to encourage the refusal of out-of-knowledge questions. Experimental results on mathematical questions affirm the substantial efficacy of RLKF in significantly enhancing LLM reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04303v1">Trans-Tokenization and Cross-lingual Vocabulary Transfers: Language Adaptation of LLMs for Low-Resource NLP</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 Accepted at COLM 2024
    </div>
    <details class="paper-abstract">
      The development of monolingual language models for low and mid-resource languages continues to be hindered by the difficulty in sourcing high-quality training data. In this study, we present a novel cross-lingual vocabulary transfer strategy, trans-tokenization, designed to tackle this challenge and enable more efficient language adaptation. Our approach focuses on adapting a high-resource monolingual LLM to an unseen target language by initializing the token embeddings of the target language using a weighted average of semantically similar token embeddings from the source language. For this, we leverage a translation resource covering both the source and target languages. We validate our method with the Tweeties, a series of trans-tokenized LLMs, and demonstrate their competitive performance on various downstream tasks across a small but diverse set of languages. Additionally, we introduce Hydra LLMs, models with multiple swappable language modeling heads and embedding tables, which further extend the capabilities of our trans-tokenization strategy. By designing a Hydra LLM based on the multilingual model TowerInstruct, we developed a state-of-the-art machine translation model for Tatar, in a zero-shot manner, completely bypassing the need for high-quality parallel data. This breakthrough is particularly significant for low-resource languages like Tatar, where high-quality parallel data is hard to come by. By lowering the data and time requirements for training high-quality models, our trans-tokenization strategy allows for the development of LLMs for a wider range of languages, especially those with limited resources. We hope that our work will inspire further research and collaboration in the field of cross-lingual vocabulary transfer and contribute to the empowerment of languages on a global scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04293v1">Are Social Sentiments Inherent in LLMs? An Empirical Study on Extraction of Inter-demographic Sentiments</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are supposed to acquire unconscious human knowledge and feelings, such as social common sense and biases, by training models from large amounts of text. However, it is not clear how much the sentiments of specific social groups can be captured in various LLMs. In this study, we focus on social groups defined in terms of nationality, religion, and race/ethnicity, and validate the extent to which sentiments between social groups can be captured in and extracted from LLMs. Specifically, we input questions regarding sentiments from one group to another into LLMs, apply sentiment analysis to the responses, and compare the results with social surveys. The validation results using five representative LLMs showed higher correlations with relatively small p-values for nationalities and religions, whose number of data points were relatively large. This result indicates that the LLM responses including the inter-group sentiments align well with actual social survey results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04237v1">Learning to Rewrite: Generalized LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be abused at scale to create non-factual content and spread disinformation. Detecting LLM-generated content is essential to mitigate these risks, but current classifiers often fail to generalize in open-world contexts. Prior work shows that LLMs tend to rewrite LLM-generated content less frequently, which can be used for detection and naturally generalizes to unforeseen data. However, we find that the rewriting edit distance between human and LLM content can be indistinguishable across domains, leading to detection failures. We propose training an LLM to rewrite input text, producing minimal edits for LLM-generated content and more edits for human-written text, deriving a distinguishable and generalizable edit distance difference across different domains. Experiments on text from 21 independent domains and three popular LLMs (e.g., GPT-4o, Gemini, and Llama-3) show that our classifier outperforms the state-of-the-art zero-shot classifier by up to 20.6% on AUROC score and the rewriting classifier by 9.2% on F1 score. Our work suggests that LLM can effectively detect machine-generated text if they are trained properly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04682v1">ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) advancements sparked a growing research interest in tool assisted LLMs solving real-world challenges, which calls for comprehensive evaluation of tool-use capabilities. While previous works focused on either evaluating over stateless web services (RESTful API), based on a single turn user prompt, or an off-policy dialog trajectory, ToolSandbox includes stateful tool execution, implicit state dependencies between tools, a built-in user simulator supporting on-policy conversational evaluation and a dynamic evaluation strategy for intermediate and final milestones over an arbitrary trajectory. We show that open source and proprietary models have a significant performance gap, and complex tasks like State Dependency, Canonicalization and Insufficient Information defined in ToolSandbox are challenging even the most capable SOTA LLMs, providing brand-new insights into tool-use LLM capabilities. ToolSandbox evaluation framework is released at https://github.com/apple/ToolSandbox
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04223v1">VideoQA in the Era of LLMs: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video-LLMs) are flourishing and has advanced many video-language tasks. As a golden testbed, Video Question Answering (VideoQA) plays pivotal role in Video-LLM developing. This work conducts a timely and comprehensive study of Video-LLMs' behavior in VideoQA, aiming to elucidate their success and failure modes, and provide insights towards more human-like video understanding and question answering. Our analyses demonstrate that Video-LLMs excel in VideoQA; they can correlate contextual cues and generate plausible responses to questions about varied video contents. However, models falter in handling video temporality, both in reasoning about temporal content ordering and grounding QA-relevant temporal moments. Moreover, the models behave unintuitively - they are unresponsive to adversarial video perturbations while being sensitive to simple variations of candidate answers and questions. Also, they do not necessarily generalize better. The findings demonstrate Video-LLMs' QA capability in standard condition yet highlight their severe deficiency in robustness and interpretability, suggesting the urgent need on rationales in Video-LLM developing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04217v1">Simplifying Translations for Children: Iterative Simplification Considering Age of Acquisition with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 Findings of ACL 2024
    </div>
    <details class="paper-abstract">
      In recent years, neural machine translation (NMT) has been widely used in everyday life. However, the current NMT lacks a mechanism to adjust the difficulty level of translations to match the user's language level. Additionally, due to the bias in the training data for NMT, translations of simple source sentences are often produced with complex words. In particular, this could pose a problem for children, who may not be able to understand the meaning of the translations correctly. In this study, we propose a method that replaces words with high Age of Acquisitions (AoA) in translations with simpler words to match the translations to the user's level. We achieve this by using large language models (LLMs), providing a triple of a source sentence, a translation, and a target word to be replaced. We create a benchmark dataset using back-translation on Simple English Wikipedia. The experimental results obtained from the dataset show that our method effectively replaces high-AoA words with lower-AoA words and, moreover, can iteratively replace most of the high-AoA words while still maintaining high BLEU and COMET scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06373v4">LLM Discussion: Enhancing the Creativity of Large Language Models via Discussion Framework and Role-Play</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 40 pages, 9 figures, COLM 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown exceptional proficiency in natural language processing but often fall short of generating creative and original responses to open-ended questions. To enhance LLM creativity, our key insight is to emulate the human process of inducing collective creativity through engaging discussions with participants from diverse backgrounds and perspectives. To this end, we propose LLM Discussion, a three-phase discussion framework that facilitates vigorous and diverging idea exchanges and ensures convergence to creative answers. Moreover, we adopt a role-playing technique by assigning distinct roles to LLMs to combat the homogeneity of LLMs. We evaluate the efficacy of the proposed framework with the Alternative Uses Test, Similarities Test, Instances Test, and Scientific Creativity Test through both LLM evaluation and human study. The results show that our proposed framework outperforms single-LLM approaches and existing multi-LLM frameworks across various creativity metrics. The code is available at https://github.com/lawraa/LLM-Discussion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04211v1">MMREC: LLM Based Multi-Modal Recommender System</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      The importance of recommender systems is growing rapidly due to the exponential increase in the volume of content generated daily. This surge in content presents unique challenges for designing effective recommender systems. Key among these challenges is the need to effectively leverage the vast amounts of natural language data and images that represent user preferences. This paper presents a novel approach to enhancing recommender systems by leveraging Large Language Models (LLMs) and deep learning techniques. The proposed framework aims to improve the accuracy and relevance of recommendations by incorporating multi-modal information processing and by the use of unified latent space representation. The study explores the potential of LLMs to better understand and utilize natural language data in recommendation contexts, addressing the limitations of previous methods. The framework efficiently extracts and integrates text and image information through LLMs, unifying diverse modalities in a latent space to simplify the learning process for the ranking model. Experimental results demonstrate the enhanced discriminative power of the model when utilizing multi-modal information. This research contributes to the evolving field of recommender systems by showcasing the potential of LLMs and multi-modal data integration to create more personalized and contextually relevant recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03528v2">Exploring the extent of similarities in software failures across industries using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      The rapid evolution of software development necessitates enhanced safety measures. Extracting information about software failures from companies is becoming increasingly more available through news articles. This research utilizes the Failure Analysis Investigation with LLMs (FAIL) model to extract industry-specific information. Although the FAIL model's database is rich in information, it could benefit from further categorization and industry-specific insights to further assist software engineers. In previous work news articles were collected from reputable sources and categorized by incidents inside a database. Prompt engineering and Large Language Models (LLMs) were then applied to extract relevant information regarding the software failure. This research extends these methods by categorizing articles into specific domains and types of software failures. The results are visually represented through graphs. The analysis shows that throughout the database some software failures occur significantly more often in specific industries. This categorization provides a valuable resource for software engineers and companies to identify and address common failures. This research highlights the synergy between software engineering and Large Language Models (LLMs) to automate and enhance the analysis of software failures. By transforming data from the database into an industry specific model, we provide a valuable resource that can be used to identify common vulnerabilities, predict potential risks, and implement proactive measures for preventing software failures. Leveraging the power of the current FAIL database and data visualization, we aim to provide an avenue for safer and more secure software in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06555v3">LLMs Learn Task Heuristics from Demonstrations: A Heuristic-Driven Prompting Strategy for Document-Level Event Argument Extraction</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 Accepted to ACL 2024
    </div>
    <details class="paper-abstract">
      In this study, we investigate in-context learning (ICL) in document-level event argument extraction (EAE) to alleviate the dependency on large-scale labeled data for this task. We introduce the Heuristic-Driven Link-of-Analogy (HD-LoA) prompting to address the challenge of example selection and to develop a prompting strategy tailored for EAE. Specifically, we hypothesize and validate that LLMs learn task-specific heuristics from demonstrations via ICL. Building upon this hypothesis, we introduce an explicit heuristic-driven demonstration construction approach, which transforms the haphazard example selection process into a methodical method that emphasizes task heuristics. Additionally, inspired by the analogical reasoning of human, we propose the link-of-analogy prompting, which enables LLMs to process new situations by drawing analogies to known situations, enhancing their performance on unseen classes beyond limited ICL examples. Experiments show that our method outperforms existing prompting methods and few-shot supervised learning methods on document-level EAE datasets. Additionally, the HD-LoA prompting shows effectiveness in diverse tasks like sentiment analysis and natural language inference, demonstrating its broad adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03675v2">NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 Accepted by ACL 2024 (main conference, long paper)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have ignited an innovative surge of AI applications, marking a new era of exciting possibilities equipped with extended context windows. However, hosting these models is cost-prohibitive mainly due to the extensive memory consumption of KV Cache involving long-context modeling. Despite several works proposing to evict unnecessary tokens from the KV Cache, most of them rely on the biased local statistics of accumulated attention scores and report performance using unconvincing metric like perplexity on inadequate short-text evaluation. In this paper, we propose NACL, a general framework for long-context KV cache eviction that achieves more optimal and efficient eviction in a single operation during the encoding phase. Due to NACL's efficiency, we combine more accurate attention score statistics in PROXY TOKENS EVICTION with the diversified random eviction strategy of RANDOM EVICTION, aiming to alleviate the issue of attention bias and enhance the robustness in maintaining pivotal tokens for long-context modeling tasks. Notably, our method significantly improves the performance on short- and long-text tasks by 80% and 76% respectively, reducing KV Cache by up to 50% with over 95% performance maintenance. The code is available at https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04121v1">Can Rule-Based Insights Enhance LLMs for Radiology Report Classification? Introducing the RadPrompt Methodology</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 Accepted at BioNLP, ACL 2024
    </div>
    <details class="paper-abstract">
      Developing imaging models capable of detecting pathologies from chest X-rays can be cost and time-prohibitive for large datasets as it requires supervision to attain state-of-the-art performance. Instead, labels extracted from radiology reports may serve as distant supervision since these are routinely generated as part of clinical practice. Despite their widespread use, current rule-based methods for label extraction rely on extensive rule sets that are limited in their robustness to syntactic variability. To alleviate these limitations, we introduce RadPert, a rule-based system that integrates an uncertainty-aware information schema with a streamlined set of rules, enhancing performance. Additionally, we have developed RadPrompt, a multi-turn prompting strategy that leverages RadPert to bolster the zero-shot predictive capabilities of large language models, achieving a statistically significant improvement in weighted average F1 score over GPT-4 Turbo. Most notably, RadPrompt surpasses both its underlying models, showcasing the synergistic potential of LLMs with rule-based models. We have evaluated our methods on two English Corpora: the MIMIC-CXR gold-standard test set and a gold-standard dataset collected from the Cambridge University Hospitals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04112v1">Patchview: LLM-Powered Worldbuilding with Generative Dust and Magnet Visualization</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 Accepted to UIST2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can help writers build story worlds by generating world elements, such as factions, characters, and locations. However, making sense of many generated elements can be overwhelming. Moreover, if the user wants to precisely control aspects of generated elements that are difficult to specify verbally, prompting alone may be insufficient. We introduce Patchview, a customizable LLM-powered system that visually aids worldbuilding by allowing users to interact with story concepts and elements through the physical metaphor of magnets and dust. Elements in Patchview are visually dragged closer to concepts with high relevance, facilitating sensemaking. The user can also steer the generation with verbally elusive concepts by indicating the desired position of the element between concepts. When the user disagrees with the LLM's visualization and generation, they can correct those by repositioning the element. These corrections can be used to align the LLM's future behaviors to the user's perception. With a user study, we show that Patchview supports the sensemaking of world elements and steering of element generation, facilitating exploration during the worldbuilding process. Patchview provides insights on how customizable visual representation can help sensemake, steer, and align generative AI model behaviors with the user's intentions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04107v1">Zero-Delay QKV Compression for Mitigating KV Cache and Network Bottlenecks in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      In large-language models, memory constraints in the key-value cache (KVC) pose a challenge during inference, especially with long prompts. In this work, we observed that compressing KV values is more effective than compressing the model regarding accuracy and job completion time (JCT). However, quantizing KV values and dropping less-important tokens incur significant runtime computational time overhead, delaying JCT. These methods also cannot reduce computation time or high network communication time overhead in sequence-parallelism (SP) frameworks for long prompts. To tackle these issues, based on our insightful observations from experimental analysis, we propose ZeroC, a Zero-delay QKV Compression system that eliminates time overhead and even reduces computation and communication time of the model operations. ZeroC innovatively embeds compression and decompression operations within model operations and adaptively determines compression ratios at a hybrid layer-token level. Further, it enables a communication-efficient SP inference framework. Trace-driven experiments demonstrate that ZeroC achieves up to 80% lower average JCT, 35% lower average perplexity, and 2.8x higher throughput with the same latency compared to state-of-the-art compression methods. ZeroC also reduces the average JCT of current LLM serving systems by up to 91% with the constraint of 0.1 perplexity increase. We open-sourced the code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.16712v2">LLM Performance Predictors are good initializers for Architecture Search</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      In this work, we utilize Large Language Models (LLMs) for a novel use case: constructing Performance Predictors (PP) that estimate the performance of specific deep neural network architectures on downstream tasks. We create PP prompts for LLMs, comprising (i) role descriptions, (ii) instructions for the LLM, (iii) hyperparameter definitions, and (iv) demonstrations presenting sample architectures with efficiency metrics and `training from scratch' performance. In machine translation (MT) tasks, GPT-4 with our PP prompts (LLM-PP) achieves a SoTA mean absolute error and a slight degradation in rank correlation coefficient compared to baseline predictors. Additionally, we demonstrate that predictions from LLM-PP can be distilled to a compact regression model (LLM-Distill-PP), which surprisingly retains much of the performance of LLM-PP. This presents a cost-effective alternative for resource-intensive performance estimation. Specifically, for Neural Architecture Search (NAS), we introduce a Hybrid-Search algorithm (HS-NAS) employing LLM-Distill-PP for the initial search stages and reverting to the baseline predictor later. HS-NAS performs similarly to SoTA NAS, reducing search hours by approximately 50%, and in some cases, improving latency, GFLOPs, and model size. The code can be found at: https://github.com/UBC-NLP/llmas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16816v2">IndicGenBench: A Multilingual Benchmark to Evaluate Generation Capabilities of LLMs on Indic Languages</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 ACL 2024
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) see increasing adoption across the globe, it is imperative for LLMs to be representative of the linguistic diversity of the world. India is a linguistically diverse country of 1.4 Billion people. To facilitate research on multilingual LLM evaluation, we release IndicGenBench - the largest benchmark for evaluating LLMs on user-facing generation tasks across a diverse set 29 of Indic languages covering 13 scripts and 4 language families. IndicGenBench is composed of diverse generation tasks like cross-lingual summarization, machine translation, and cross-lingual question answering. IndicGenBench extends existing benchmarks to many Indic languages through human curation providing multi-way parallel evaluation data for many under-represented Indic languages for the first time. We evaluate a wide range of proprietary and open-source LLMs including GPT-3.5, GPT-4, PaLM-2, mT5, Gemma, BLOOM and LLaMA on IndicGenBench in a variety of settings. The largest PaLM-2 models performs the best on most tasks, however, there is a significant performance gap in all languages compared to English showing that further research is needed for the development of more inclusive multilingual language models. IndicGenBench is released at www.github.com/google-research-datasets/indic-gen-bench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02964v2">Accuracy and Consistency of LLMs in the Registered Dietitian Exam: The Impact of Prompt Engineering and Knowledge Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are fundamentally transforming human-facing applications in the health and well-being domains: boosting patient engagement, accelerating clinical decision-making, and facilitating medical education. Although state-of-the-art LLMs have shown superior performance in several conversational applications, evaluations within nutrition and diet applications are still insufficient. In this paper, we propose to employ the Registered Dietitian (RD) exam to conduct a standard and comprehensive evaluation of state-of-the-art LLMs, GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro, assessing both accuracy and consistency in nutrition queries. Our evaluation includes 1050 RD exam questions encompassing several nutrition topics and proficiency levels. In addition, for the first time, we examine the impact of Zero-Shot (ZS), Chain of Thought (CoT), Chain of Thought with Self Consistency (CoT-SC), and Retrieval Augmented Prompting (RAP) on both accuracy and consistency of the responses. Our findings revealed that while these LLMs obtained acceptable overall performance, their results varied considerably with different prompts and question domains. GPT-4o with CoT-SC prompting outperformed the other approaches, whereas Gemini 1.5 Pro with ZS recorded the highest consistency. For GPT-4o and Claude 3.5, CoT improved the accuracy, and CoT-SC improved both accuracy and consistency. RAP was particularly effective for GPT-4o to answer Expert level questions. Consequently, choosing the appropriate LLM and prompting technique, tailored to the proficiency level and specific domain, can mitigate errors and potential risks in diet and nutrition chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04023v1">Improving Large Language Model (LLM) fidelity through context-aware grounding: A systematic approach to reliability and veracity</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly sophisticated and ubiquitous in natural language processing (NLP) applications, ensuring their robustness, trustworthiness, and alignment with human values has become a critical challenge. This paper presents a novel framework for contextual grounding in textual models, with a particular emphasis on the Context Representation stage. Our approach aims to enhance the reliability and ethical alignment of these models through a comprehensive, context-aware methodology. By explicitly capturing and representing relevant situational, cultural, and ethical contexts in a machine-readable format, we lay the foundation for anchoring a model's behavior within these contexts. Our approach leverages techniques from knowledge representation and reasoning, such as ontologies, semantic web technologies, and logic-based formalisms. We evaluate our framework on real-world textual datasets, demonstrating its effectiveness in improving model performance, fairness, and alignment with human expectations, while maintaining high accuracy. Furthermore, we discuss the other key components of the framework, including context-aware encoding, context-aware learning, interpretability and explainability, and continuous monitoring and adaptation. This research contributes to the growing body of work on responsible AI, offering a practical approach to developing more reliable, trustworthy, and ethically-aligned language models. Our findings have significant implications for the deployment of LLMs in sensitive domains such as healthcare, legal systems, and social services, where contextual understanding is paramount.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03907v1">Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 6 pages paper content, 17 pages of appendix
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have excelled at language understanding and generating human-level text. However, even with supervised training and human alignment, these LLMs are susceptible to adversarial attacks where malicious users can prompt the model to generate undesirable text. LLMs also inherently encode potential biases that can cause various harmful effects during interactions. Bias evaluation metrics lack standards as well as consensus and existing methods often rely on human-generated templates and annotations which are expensive and labor intensive. In this work, we train models to automatically create adversarial prompts to elicit biased responses from target LLMs. We present LLM- based bias evaluation metrics and also analyze several existing automatic evaluation methods and metrics. We analyze the various nuances of model responses, identify the strengths and weaknesses of model families, and assess where evaluation methods fall short. We compare these metrics to human evaluation and validate that the LLM-as-a-Judge metric aligns with human judgement on bias in response generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03876v1">From Data to Story: Towards Automatic Animated Data Video Creation with LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 6 pages, 2 figures, IEEE VIS 2024 Gen4DS Workshop
    </div>
    <details class="paper-abstract">
      Creating data stories from raw data is challenging due to humans' limited attention spans and the need for specialized skills. Recent advancements in large language models (LLMs) offer great opportunities to develop systems with autonomous agents to streamline the data storytelling workflow. Though multi-agent systems have benefits such as fully realizing LLM potentials with decomposed tasks for individual agents, designing such systems also faces challenges in task decomposition, performance optimization for sub-tasks, and workflow design. To better understand these issues, we develop Data Director, an LLM-based multi-agent system designed to automate the creation of animated data videos, a representative genre of data stories. Data Director interprets raw data, breaks down tasks, designs agent roles to make informed decisions automatically, and seamlessly integrates diverse components of data videos. A case study demonstrates Data Director's effectiveness in generating data videos. Throughout development, we have derived lessons learned from addressing challenges, guiding further advancements in autonomous agents for data storytelling. We also shed light on future directions for global optimization, human-in-the-loop design, and the application of advanced multi-modal LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.07705v3">CA-LoRA: Adapting Existing LoRA for Compressed LLMs to Enable Efficient Multi-Tasking on Personal Devices</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      Recently, there has been a demand to deploy Large Language Models (LLMs) on personal devices such as laptops and smartphones. These LLMs have different model variants when handling different tasks. However, personal devices have limited resources and require reduced storage overhead. To address this, there are two key methods available: the first is model compression, which compresses LLMs into smaller sizes; the second is LoRA, which can transfer an LLM to other tasks with very few parameters, avoiding the storage of multiple model variants in multi-task scenarios by only preserving LoRAs. However, our experiments show that directly combining these two methods yields sub-optimal performance. Considering that the open-source community has already contributed many LoRAs to LLMs, we propose to adapt these existing LoRAs from the LLMs to their compressed version and introduce a Compression-Aware LoRA (CA-LoRA) framework. We incorporate knowledge inheritance and recovery strategies to recover the lost knowledge caused by model compression. Experiment results demonstrate that CA-LoRA outperforms the vanilla LoRA methods applied to a compressed LLM and achieves comparable performance to the non-compressed LLM with existing LoRA modules. The source code of CA-LoRA is available at https://github.com/thunlp/CA-LoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18990v2">Stay Tuned: An Empirical Study of the Impact of Hyperparameters on LLM Tuning in Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) is an effective method to enhance their performance on downstream tasks. However, choosing the appropriate setting of tuning hyperparameters (HPs) is a labor-intensive and computationally expensive process. Here, we provide recommended HP configurations for practical use-cases that represent a better starting point for practitioners, when considering two SOTA LLMs and two commonly used tuning methods. We describe Coverage-based Search (CBS), a process for ranking HP configurations based on an offline extensive grid search, such that the top ranked configurations collectively provide a practical robust recommendation for a wide range of datasets and domains. We focus our experiments on Llama-3-8B and Mistral-7B, as well as full fine-tuning and LoRa, conducting a total of > 10,000 tuning experiments. Our results suggest that, in general, Llama-3-8B and LoRA should be preferred, when possible. Moreover, we show that for both models and tuning methods, exploring only a few HP configurations, as recommended by our analysis, can provide excellent results in practice, making this work a valuable resource for practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11755v3">Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 ECCV Camera Ready. Code & Data: https://jmiemirza.github.io/Meta-Prompting/
    </div>
    <details class="paper-abstract">
      Prompt ensembling of Large Language Model (LLM) generated category-specific prompts has emerged as an effective method to enhance zero-shot recognition ability of Vision-Language Models (VLMs). To obtain these category-specific prompts, the present methods rely on hand-crafting the prompts to the LLMs for generating VLM prompts for the downstream tasks. However, this requires manually composing these task-specific prompts and still, they might not cover the diverse set of visual concepts and task-specific styles associated with the categories of interest. To effectively take humans out of the loop and completely automate the prompt generation process for zero-shot recognition, we propose Meta-Prompting for Visual Recognition (MPVR). Taking as input only minimal information about the target task, in the form of its short natural language description, and a list of associated class labels, MPVR automatically produces a diverse set of category-specific prompts resulting in a strong zero-shot classifier. MPVR generalizes effectively across various popular zero-shot image recognition benchmarks belonging to widely different domains when tested with multiple LLMs and VLMs. For example, MPVR obtains a zero-shot recognition improvement over CLIP by up to 19.8% and 18.2% (5.0% and 4.5% on average over 20 datasets) leveraging GPT and Mixtral LLMs, respectively
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03562v1">A Comparison of LLM Finetuning Methods & Evaluation Metrics with Travel Chatbot Use Case</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      This research compares large language model (LLM) fine-tuning methods, including Quantized Low Rank Adapter (QLoRA), Retrieval Augmented fine-tuning (RAFT), and Reinforcement Learning from Human Feedback (RLHF), and additionally compared LLM evaluation methods including End to End (E2E) benchmark method of "Golden Answers", traditional natural language processing (NLP) metrics, RAG Assessment (Ragas), OpenAI GPT-4 evaluation metrics, and human evaluation, using the travel chatbot use case. The travel dataset was sourced from the the Reddit API by requesting posts from travel-related subreddits to get travel-related conversation prompts and personalized travel experiences, and augmented for each fine-tuning method. We used two pretrained LLMs utilized for fine-tuning research: LLaMa 2 7B, and Mistral 7B. QLoRA and RAFT are applied to the two pretrained models. The inferences from these models are extensively evaluated against the aforementioned metrics. The best model according to human evaluation and some GPT-4 metrics was Mistral RAFT, so this underwent a Reinforcement Learning from Human Feedback (RLHF) training pipeline, and ultimately was evaluated as the best model. Our main findings are that: 1) quantitative and Ragas metrics do not align with human evaluation, 2) Open AI GPT-4 evaluation most aligns with human evaluation, 3) it is essential to keep humans in the loop for evaluation because, 4) traditional NLP metrics insufficient, 5) Mistral generally outperformed LLaMa, 6) RAFT outperforms QLoRA, but still needs postprocessing, 7) RLHF improves model performance significantly. Next steps include improving data quality, increasing data quantity, exploring RAG methods, and focusing data collection on a specific city, which would improve data quality by narrowing the focus, while creating a useful product.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03561v1">MPC-Minimized Secure LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      Many inference services based on large language models (LLMs) pose a privacy concern, either revealing user prompts to the service or the proprietary weights to the user. Secure inference offers a solution to this problem through secure multi-party computation (MPC), however, it is still impractical for modern LLM workload due to the large overhead imposed by MPC. To address this overhead, we propose Marill, a framework that adapts LLM fine-tuning to minimize MPC usage during secure inference. Marill introduces high-level architectural changes during fine-tuning that significantly reduce the number of expensive operations needed within MPC during inference, by removing some and relocating others outside MPC without compromising security. As a result, Marill-generated models are more efficient across all secure inference protocols and our approach complements MPC-friendly approximations for such operations. Compared to standard fine-tuning, Marill results in 3.6-11.3x better runtime and 2.4-6.9x better communication during secure inference across various MPC settings, while typically preserving over 90% performance across downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21320v2">MetaOpenFOAM: an LLM-based multi-agent framework for CFD</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 31 pages,11 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Remarkable progress has been made in automated problem solving through societies of agents based on large language models (LLMs). Computational fluid dynamics (CFD), as a complex problem, presents unique challenges in automated simulations that require sophisticated solutions. MetaOpenFOAM, as a novel multi-agent collaborations framework, aims to complete CFD simulation tasks with only natural language as input. These simulation tasks include mesh pre-processing, simulation and so on. MetaOpenFOAM harnesses the power of MetaGPT's assembly line paradigm, which assigns diverse roles to various agents, efficiently breaking down complex CFD tasks into manageable subtasks. Langchain further complements MetaOpenFOAM by integrating Retrieval-Augmented Generation (RAG) technology, which enhances the framework's ability by integrating a searchable database of OpenFOAM tutorials for LLMs. Tests on a benchmark for natural language-based CFD solver, consisting of eight CFD simulation tasks, have shown that MetaOpenFOAM achieved a high pass rate per test (85%), with each test case costing only $0.22 on average. The eight CFD simulation tasks encompass a range of multidimensional flow problems, covering compressible and incompressible flows with different physical processes. This demonstrates the capability to automate CFD simulations using only natural language input, iteratively correcting errors to achieve the desired simulations. An ablation study was conducted to verify the necessity of each component in the multi-agent system and the RAG technology. A sensitivity study on the randomness of LLM showed that LLM with low randomness can obtain more stable and accurate results. Additionally, MetaOpenFOAM owns the ability to identify and modify key parameters in user requirements, and excels in correcting bugs when failure match occur,which demonstrates the generalization of MetaOpenFOAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03505v1">Optimus: Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have extended the success of large language models (LLMs) to multiple data types, such as image, text and audio, achieving significant performance in various domains, including multimodal translation, visual question answering and content generation. Nonetheless, existing systems are inefficient to train MLLMs due to substantial GPU bubbles caused by the heterogeneous modality models and complex data dependencies in 3D parallelism. This paper proposes Optimus, a distributed MLLM training system that reduces end-to-end MLLM training time. Optimus is based on our principled analysis that scheduling the encoder computation within the LLM bubbles can reduce bubbles in MLLM training. To make scheduling encoder computation possible for all GPUs, Optimus searches the separate parallel plans for encoder and LLM, and adopts a bubble scheduling algorithm to enable exploiting LLM bubbles without breaking the original data dependencies in the MLLM model architecture. We further decompose encoder layer computation into a series of kernels, and analyze the common bubble pattern of 3D parallelism to carefully optimize the sub-millisecond bubble scheduling, minimizing the overall training time. Our experiments in a production cluster show that Optimus accelerates MLLM training by 20.5%-21.3% with ViT-22B and GPT-175B model over 3072 GPUs compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00114v2">Inductive or Deductive? Rethinking the Fundamental Reasoning Abilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      Reasoning encompasses two typical types: deductive reasoning and inductive reasoning. Despite extensive research into the reasoning capabilities of Large Language Models (LLMs), most studies have failed to rigorously differentiate between inductive and deductive reasoning, leading to a blending of the two. This raises an essential question: In LLM reasoning, which poses a greater challenge - deductive or inductive reasoning? While the deductive reasoning capabilities of LLMs, (i.e. their capacity to follow instructions in reasoning tasks), have received considerable attention, their abilities in true inductive reasoning remain largely unexplored. To investigate into the true inductive reasoning capabilities of LLMs, we propose a novel framework, SolverLearner. This framework enables LLMs to learn the underlying function (i.e., $y = f_w(x)$), that maps input data points $(x)$ to their corresponding output values $(y)$, using only in-context examples. By focusing on inductive reasoning and separating it from LLM-based deductive reasoning, we can isolate and investigate inductive reasoning of LLMs in its pure form via SolverLearner. Our observations reveal that LLMs demonstrate remarkable inductive reasoning capabilities through SolverLearner, achieving near-perfect performance with ACC of 1 in most cases. Surprisingly, despite their strong inductive reasoning abilities, LLMs tend to relatively lack deductive reasoning capabilities, particularly in tasks involving ``counterfactual'' reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03489v1">Harnessing the Power of LLMs in Source Code Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      Software vulnerabilities, caused by unintentional flaws in source code, are a primary root cause of cyberattacks. Static analysis of source code has been widely used to detect these unintentional defects introduced by software developers. Large Language Models (LLMs) have demonstrated human-like conversational abilities due to their capacity to capture complex patterns in sequential data, such as natural languages. In this paper, we harness LLMs' capabilities to analyze source code and detect known vulnerabilities. To ensure the proposed vulnerability detection method is universal across multiple programming languages, we convert source code to LLVM IR and train LLMs on these intermediate representations. We conduct extensive experiments on various LLM architectures and compare their accuracy. Our comprehensive experiments on real-world and synthetic codes from NVD and SARD demonstrate high accuracy in identifying source code vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03475v1">Can LLMs Serve As Time Series Anomaly Detectors?</a></div>
    <div class="paper-meta">
      📅 2024-08-06
    </div>
    <details class="paper-abstract">
      An emerging topic in large language models (LLMs) is their application to time series forecasting, characterizing mainstream and patternable characteristics of time series. A relevant but rarely explored and more challenging question is whether LLMs can detect and explain time series anomalies, a critical task across various real-world applications. In this paper, we investigate the capabilities of LLMs, specifically GPT-4 and LLaMA3, in detecting and explaining anomalies in time series. Our studies reveal that: 1) LLMs cannot be directly used for time series anomaly detection. 2) By designing prompt strategies such as in-context learning and chain-of-thought prompting, GPT-4 can detect time series anomalies with results competitive to baseline methods. 3) We propose a synthesized dataset to automatically generate time series anomalies with corresponding explanations. By applying instruction fine-tuning on this dataset, LLaMA3 demonstrates improved performance in time series anomaly detection tasks. In summary, our exploration shows the promising potential of LLMs as time series anomaly detectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03408v1">LLM-Aided Compilation for Tensor Accelerators</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 4 page workshop paper
    </div>
    <details class="paper-abstract">
      Hardware accelerators, in particular accelerators for tensor processing, have many potential application domains. However, they currently lack the software infrastructure to support the majority of domains outside of deep learning. Furthermore, a compiler that can easily be updated to reflect changes at both application and hardware levels would enable more agile development and design space exploration of accelerators, allowing hardware designers to realize closer-to-optimal performance. In this work, we discuss how large language models (LLMs) could be leveraged to build such a compiler. Specifically, we demonstrate the ability of GPT-4 to achieve high pass rates in translating code to the Gemmini accelerator, and prototype a technique for decomposing translation into smaller, more LLM-friendly steps. Additionally, we propose a 2-phase workflow for utilizing LLMs to generate hardware-optimized code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03319v1">Training LLMs to Recognize Hedges in Spontaneous Narratives</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 Amie Paige, Adil Soubki, and John Murzaku contributed equally to this study
    </div>
    <details class="paper-abstract">
      Hedges allow speakers to mark utterances as provisional, whether to signal non-prototypicality or "fuzziness", to indicate a lack of commitment to an utterance, to attribute responsibility for a statement to someone else, to invite input from a partner, or to soften critical feedback in the service of face-management needs. Here we focus on hedges in an experimentally parameterized corpus of 63 Roadrunner cartoon narratives spontaneously produced from memory by 21 speakers for co-present addressees, transcribed to text (Galati and Brennan, 2010). We created a gold standard of hedges annotated by human coders (the Roadrunner-Hedge corpus) and compared three LLM-based approaches for hedge detection: fine-tuning BERT, and zero and few-shot prompting with GPT-4o and LLaMA-3. The best-performing approach was a fine-tuned BERT model, followed by few-shot GPT-4o. After an error analysis on the top performing approaches, we used an LLM-in-the-Loop approach to improve the gold standard coding, as well as to highlight cases in which hedges are ambiguous in linguistically interesting ways that will guide future research. This is the first step in our research program to train LLMs to interpret and generate collateral signals appropriately and meaningfully in conversation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03314v1">Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters</a></div>
    <div class="paper-meta">
      📅 2024-08-06
    </div>
    <details class="paper-abstract">
      Enabling LLMs to improve their outputs by using more test-time computation is a critical step towards building generally self-improving agents that can operate on open-ended natural language. In this paper, we study the scaling of inference-time computation in LLMs, with a focus on answering the question: if an LLM is allowed to use a fixed but non-trivial amount of inference-time compute, how much can it improve its performance on a challenging prompt? Answering this question has implications not only on the achievable performance of LLMs, but also on the future of LLM pretraining and how one should tradeoff inference-time and pre-training compute. Despite its importance, little research attempted to understand the scaling behaviors of various test-time inference methods. Moreover, current work largely provides negative results for a number of these strategies. In this work, we analyze two primary mechanisms to scale test-time computation: (1) searching against dense, process-based verifier reward models; and (2) updating the model's distribution over a response adaptively, given the prompt at test time. We find that in both cases, the effectiveness of different approaches to scaling test-time compute critically varies depending on the difficulty of the prompt. This observation motivates applying a "compute-optimal" scaling strategy, which acts to most effectively allocate test-time compute adaptively per prompt. Using this compute-optimal strategy, we can improve the efficiency of test-time compute scaling by more than 4x compared to a best-of-N baseline. Additionally, in a FLOPs-matched evaluation, we find that on problems where a smaller base model attains somewhat non-trivial success rates, test-time compute can be used to outperform a 14x larger model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04666v1">LLMs are Not Just Next Token Predictors</a></div>
    <div class="paper-meta">
      📅 2024-08-06
    </div>
    <details class="paper-abstract">
      LLMs are statistical models of language learning through stochastic gradient descent with a next token prediction objective. Prompting a popular view among AI modelers: LLMs are just next token predictors. While LLMs are engineered using next token prediction, and trained based on their success at this task, our view is that a reduction to just next token predictor sells LLMs short. Moreover, there are important explanations of LLM behavior and capabilities that are lost when we engage in this kind of reduction. In order to draw this out, we will make an analogy with a once prominent research program in biology explaining evolution and development from the gene's eye view.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03256v1">Synthesizing Text-to-SQL Data from Weak and Strong LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 12 pages, 7 figures, ACL 2024
    </div>
    <details class="paper-abstract">
      The capability gap between open-source and closed-source large language models (LLMs) remains a challenge in text-to-SQL tasks. In this paper, we introduce a synthetic data approach that combines data produced by larger, more powerful models (strong models) with error information data generated by smaller, not well-aligned models (weak models). The method not only enhances the domain generalization of text-to-SQL models but also explores the potential of error data supervision through preference learning. Furthermore, we employ the synthetic data approach for instruction tuning on open-source LLMs, resulting SENSE, a specialized text-to-SQL model. The effectiveness of SENSE is demonstrated through state-of-the-art results on the SPIDER and BIRD benchmarks, bridging the performance gap between open-source models and methods prompted by closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03623v2">Unveiling LLMs: The Evolution of Latent Representations in a Dynamic Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 Accepted at COLM 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate an impressive capacity to recall a vast range of factual knowledge. However, understanding their underlying reasoning and internal mechanisms in exploiting this knowledge remains a key research area. This work unveils the factual information an LLM represents internally for sentence-level claim verification. We propose an end-to-end framework to decode factual knowledge embedded in token representations from a vector space to a set of ground predicates, showing its layer-wise evolution using a dynamic knowledge graph. Our framework employs activation patching, a vector-level technique that alters a token representation during inference, to extract encoded knowledge. Accordingly, we neither rely on training nor external models. Using factual and common-sense claims from two claim verification datasets, we showcase interpretability analyses at local and global levels. The local analysis highlights entity centrality in LLM reasoning, from claim-related information and multi-hop reasoning to representation errors causing erroneous evaluation. On the other hand, the global reveals trends in the underlying evolution, such as word-based knowledge evolving into claim-related facts. By interpreting semantics from LLM latent representations and enabling graph-related analyses, this work enhances the understanding of the factual knowledge resolution process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04665v1">LLM-based MOFs Synthesis Condition Extraction using Few-Shot Demonstrations</a></div>
    <div class="paper-meta">
      📅 2024-08-06
    </div>
    <details class="paper-abstract">
      The extraction of Metal-Organic Frameworks (MOFs) synthesis conditions from literature text has been challenging but crucial for the logical design of new MOFs with desirable functionality. The recent advent of large language models (LLMs) provides disruptively new solution to this long-standing problem and latest researches have reported over 90% F1 in extracting correct conditions from MOFs literature. We argue in this paper that most existing synthesis extraction practices with LLMs stay with the primitive zero-shot learning, which could lead to downgraded extraction and application performance due to the lack of specialized knowledge. This work pioneers and optimizes the few-shot in-context learning paradigm for LLM extraction of material synthesis conditions. First, we propose a human-AI joint data curation process to secure high-quality ground-truth demonstrations for few-shot learning. Second, we apply a BM25 algorithm based on the retrieval-augmented generation (RAG) technique to adaptively select few-shot demonstrations for each MOF's extraction. Over a dataset randomly sampled from 84,898 well-defined MOFs, the proposed few-shot method achieves much higher average F1 performance (0.93 vs. 0.81, +14.8%) than the native zero-shot LLM using the same GPT-4 model, under fully automatic evaluation that are more objective than the previous human evaluation. The proposed method is further validated through real-world material experiments: compared with the baseline zero-shot LLM, the proposed few-shot approach increases the MOFs structural inference performance (R^2) by 29.4% in average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20089v2">The Fine-Tuning Paradox: Boosting Translation Quality Without Sacrificing LLM Abilities</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 Accepted to ACL 2024 (long, main). Latest version includes link to the IdiomsInCtx-MT dataset
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for machine translation has shown improvements in overall translation quality. However, it is unclear what is the impact of fine-tuning on desirable LLM behaviors that are not present in neural machine translation models, such as steerability, inherent document-level translation abilities, and the ability to produce less literal translations. We perform an extensive translation evaluation on the LLaMA and Falcon family of models with model size ranging from 7 billion up to 65 billion parameters. Our results show that while fine-tuning improves the general translation quality of LLMs, several abilities degrade. In particular, we observe a decline in the ability to perform formality steering, to produce technical translations through few-shot examples, and to perform document-level translation. On the other hand, we observe that the model produces less literal translations after fine-tuning on parallel data. We show that by including monolingual data as part of the fine-tuning data we can maintain the abilities while simultaneously enhancing overall translation quality. Our findings emphasize the need for fine-tuning strategies that preserve the benefits of LLMs for machine translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03150v1">Conditioning LLMs with Emotion in Neural Machine Translation</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 6 pages, In Proceedings of the 21st International Conference on Spoken Language Translation (IWSLT), Bangkok, Thailand, 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance in Natural Language Processing tasks, including Machine Translation (MT). In this work, we propose a novel MT pipeline that integrates emotion information extracted from a Speech Emotion Recognition (SER) model into LLMs to enhance translation quality. We first fine-tune five existing LLMs on the Libri-trans dataset and select the most performant model. Subsequently, we augment LLM prompts with different dimensional emotions and train the selected LLM under these different configurations. Our experiments reveal that integrating emotion information, especially arousal, into LLM prompts leads to notable improvements in translation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03234v3">Self-Evaluation as a Defense Against Adversarial Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We introduce a defense against adversarial attacks on LLMs utilizing self-evaluation. Our method requires no model fine-tuning, instead using pre-trained models to evaluate the inputs and outputs of a generator model, significantly reducing the cost of implementation in comparison to other, finetuning-based methods. Our method can significantly reduce the attack success rate of attacks on both open and closed-source LLMs, beyond the reductions demonstrated by Llama-Guard2 and commonly used content moderation APIs. We present an analysis of the effectiveness of our method, including attempts to attack the evaluator in various settings, demonstrating that it is also more resilient to attacks than existing methods. Code and data will be made available at https://github.com/Linlt-leon/self-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03099v1">Topic Modeling with Fine-tuning LLMs and Bag of Sentences</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 This is the submitted journal version of enhanced with the novel fine-tuning part of "Efficient and Flexible Topic Modeling using Pretrained Embeddings and Bag of Sentences'' which appeared at the International Conference on Agents and Artificial Intelligence(ICAART) in 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLM)'s are increasingly used for topic modeling outperforming classical topic models such as LDA. Commonly, pre-trained LLM encoders such as BERT are used out-of-the-box despite the fact that fine-tuning is known to improve LLMs considerably. The challenge lies in obtaining a suitable (labeled) dataset for fine-tuning. In this paper, we use the recent idea to use bag of sentences as the elementary unit in computing topics. In turn, we derive an approach FT-Topic to perform unsupervised fine-tuning relying primarily on two steps for constructing a training dataset in an automatic fashion. First, a heuristic method to identifies pairs of sentence groups that are either assumed to be of the same or different topics. Second, we remove sentence pairs that are likely labeled incorrectly. The dataset is then used to fine-tune an encoder LLM, which can be leveraged by any topic modeling approach using embeddings. However, in this work, we demonstrate its effectiveness by deriving a novel state-of-the-art topic modeling method called SenClu, which achieves fast inference through an expectation-maximization algorithm and hard assignments of sentence groups to a single topic, while giving users the possibility to encode prior knowledge on the topic-document distribution. Code is at \url{https://github.com/JohnTailor/FT-Topic}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02999v1">LLMs as Probabilistic Minimally Adequate Teachers for DFA Learning</a></div>
    <div class="paper-meta">
      📅 2024-08-06
    </div>
    <details class="paper-abstract">
      The emergence of intelligence in large language models (LLMs) has inspired investigations into their integration into automata learning. This paper introduces the probabilistic Minimally Adequate Teacher (pMAT) formulation, which leverages a probabilistic oracle that could give persistent errors randomly during answering the membership queries for deterministic finite automata (DFA) learning. Given the tendency of LLMs to produce hallucinatory content, we have developed techniques to improve answer accuracy and ensure the correctness of the learned automata. We propose the $\mathtt{Discrimination}$ prompt as well as the $\mathtt{Verification}$ prompt and explore their advantages over common prompts. Additionally, we compare DFA learning performance between the TTT algorithm and common active learning algorithms. To address the exponential number of persistent errors, we implement a dynamic query cache refinement algorithm that identifies and corrects conflicting queries by combining the active and passive learning algorithms. The empirical results demonstrate the robustness and efficiency of our approach, providing a theoretical foundation for automata learning with LLMs in the loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02944v1">LLM-Empowered Resource Allocation in Wireless Communications Systems</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 submitted to possible IEEE journal
    </div>
    <details class="paper-abstract">
      The recent success of large language models (LLMs) has spurred their application in various fields. In particular, there have been efforts to integrate LLMs into various aspects of wireless communication systems. The use of LLMs in wireless communication systems has the potential to realize artificial general intelligence (AGI)-enabled wireless networks. In this paper, we investigate an LLM-based resource allocation scheme for wireless communication systems. Specifically, we formulate a simple resource allocation problem involving two transmit pairs and develop an LLM-based resource allocation approach that aims to maximize either energy efficiency or spectral efficiency. Additionally, we consider the joint use of low-complexity resource allocation techniques to compensate for the reliability shortcomings of the LLM-based scheme. After confirming the applicability and feasibility of LLM-based resource allocation, we address several key technical challenges that remain in applying LLMs in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02927v1">HARMONIC: Harnessing LLMs for Tabular Data Synthesis and Privacy Protection</a></div>
    <div class="paper-meta">
      📅 2024-08-06
    </div>
    <details class="paper-abstract">
      Data serves as the fundamental foundation for advancing deep learning, particularly tabular data presented in a structured format, which is highly conducive to modeling. However, even in the era of LLM, obtaining tabular data from sensitive domains remains a challenge due to privacy or copyright concerns. Hence, exploring how to effectively use models like LLMs to generate realistic and privacy-preserving synthetic tabular data is urgent. In this paper, we take a step forward to explore LLMs for tabular data synthesis and privacy protection, by introducing a new framework HARMONIC for tabular data generation and evaluation. In the tabular data generation of our framework, unlike previous small-scale LLM-based methods that rely on continued pre-training, we explore the larger-scale LLMs with fine-tuning to generate tabular data and enhance privacy. Based on idea of the k-nearest neighbors algorithm, an instruction fine-tuning dataset is constructed to inspire LLMs to discover inter-row relationships. Then, with fine-tuning, LLMs are trained to remember the format and connections of the data rather than the data itself, which reduces the risk of privacy leakage. In the evaluation part of our framework, we develop specific privacy risk metrics DLT for LLM synthetic data generation, as well as performance evaluation metrics LLE for downstream LLM tasks. Our experiments find that this tabular data generation framework achieves equivalent performance to existing methods with better privacy, which also demonstrates our evaluation framework for the effectiveness of synthetic data and privacy risks in LLM scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01964v4">Enabling Discriminative Reasoning in LLMs for Legal Judgment Prediction</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 repo: https://github.com/ChenlongDeng/ADAPT
    </div>
    <details class="paper-abstract">
      Legal judgment prediction is essential for enhancing judicial efficiency. In this work, we identify that existing large language models (LLMs) underperform in this domain due to challenges in understanding case complexities and distinguishing between similar charges. To adapt LLMs for effective legal judgment prediction, we introduce the Ask-Discriminate-Predict (ADAPT) reasoning framework inspired by human judicial reasoning. ADAPT involves decomposing case facts, discriminating among potential charges, and predicting the final judgment. We further enhance LLMs through fine-tuning with multi-task synthetic trajectories to improve legal judgment prediction accuracy and efficiency under our ADAPT framework. Extensive experiments conducted on two widely-used datasets demonstrate the superior performance of our framework in legal judgment prediction, particularly when dealing with complex and confusing charges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02861v1">A Framework for Fine-Tuning LLMs using Heterogeneous Feedback</a></div>
    <div class="paper-meta">
      📅 2024-08-05
      | 💬 7 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been applied to a wide range of tasks, including text summarization, web navigation, and chatbots. They have benefitted from supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF) following an unsupervised pretraining. These datasets can be difficult to collect, limited in scope, and vary in sample quality. Additionally, datasets can vary extensively in supervision format, from numerical to binary as well as multi-dimensional with many different values. We present a framework for fine-tuning LLMs using heterogeneous feedback, which has two main components. First, we combine the heterogeneous feedback data into a single supervision format, compatible with methods like SFT and RLHF. Next, given this unified feedback dataset, we extract a high-quality and diverse subset to obtain performance increases potentially exceeding the full dataset. We conduct extensive experiments to understand the effectiveness of these techniques for incorporating heterogeneous feedback, and demonstrate improvements from using a high-quality and diverse subset of the data. We find that our framework is able to improve models in multiple areas simultaneously, such as in instruction following and bias reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02784v1">LLM economicus? Mapping the Behavioral Biases of LLMs via Utility Theory</a></div>
    <div class="paper-meta">
      📅 2024-08-05
      | 💬 Accepted to COLM 2024
    </div>
    <details class="paper-abstract">
      Humans are not homo economicus (i.e., rational economic beings). As humans, we exhibit systematic behavioral biases such as loss aversion, anchoring, framing, etc., which lead us to make suboptimal economic decisions. Insofar as such biases may be embedded in text data on which large language models (LLMs) are trained, to what extent are LLMs prone to the same behavioral biases? Understanding these biases in LLMs is crucial for deploying LLMs to support human decision-making. We propose utility theory-a paradigm at the core of modern economic theory-as an approach to evaluate the economic biases of LLMs. Utility theory enables the quantification and comparison of economic behavior against benchmarks such as perfect rationality or human behavior. To demonstrate our approach, we quantify and compare the economic behavior of a variety of open- and closed-source LLMs. We find that the economic behavior of current LLMs is neither entirely human-like nor entirely economicus-like. We also find that most current LLMs struggle to maintain consistent economic behavior across settings. Finally, we illustrate how our approach can measure the effect of interventions such as prompting on economic biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02584v1">Leveraging the Power of LLMs: A Fine-Tuning Approach for High-Quality Aspect-Based Summarization</a></div>
    <div class="paper-meta">
      📅 2024-08-05
    </div>
    <details class="paper-abstract">
      The ever-increasing volume of digital information necessitates efficient methods for users to extract key insights from lengthy documents. Aspect-based summarization offers a targeted approach, generating summaries focused on specific aspects within a document. Despite advancements in aspect-based summarization research, there is a continuous quest for improved model performance. Given that large language models (LLMs) have demonstrated the potential to revolutionize diverse tasks within natural language processing, particularly in the problem of summarization, this paper explores the potential of fine-tuning LLMs for the aspect-based summarization task. We evaluate the impact of fine-tuning open-source foundation LLMs, including Llama2, Mistral, Gemma and Aya, on a publicly available domain-specific aspect based summary dataset. We hypothesize that this approach will enable these models to effectively identify and extract aspect-related information, leading to superior quality aspect-based summaries compared to the state-of-the-art. We establish a comprehensive evaluation framework to compare the performance of fine-tuned LLMs against competing aspect-based summarization methods and vanilla counterparts of the fine-tuned LLMs. Our work contributes to the field of aspect-based summarization by demonstrating the efficacy of fine-tuning LLMs for generating high-quality aspect-based summaries. Furthermore, it opens doors for further exploration of using LLMs for targeted information extraction tasks across various NLP domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18932v2">Be More Real: Travel Diary Generation Using LLM Agents and Individual Profiles</a></div>
    <div class="paper-meta">
      📅 2024-08-05
    </div>
    <details class="paper-abstract">
      Human mobility is inextricably linked to social issues such as traffic congestion, energy consumption, and public health; however, privacy concerns restrict access to mobility data. Recently, research have utilized Large Language Models (LLMs) for human mobility generation, in which the challenge is how LLMs can understand individuals' mobility behavioral differences to generate realistic trajectories conforming to real world contexts. This study handles this problem by presenting an LLM agent-based framework (MobAgent) composing two phases: understanding-based mobility pattern extraction and reasoning-based trajectory generation, which enables generate more real travel diaries at urban scale, considering different individual profiles. MobAgent extracts reasons behind specific mobility trendiness and attribute influences to provide reliable patterns; infers the relationships between contextual factors and underlying motivations of mobility; and based on the patterns and the recursive reasoning process, MobAgent finally generates more authentic and personalized mobilities that reflect both individual differences and real-world constraints. We validate our framework with 0.2 million travel survey data, demonstrating its effectiveness in producing personalized and accurate travel diaries. This study highlights the capacity of LLMs to provide detailed and sophisticated understanding of human mobility through the real-world mobility data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06093v2">Selective Fine-tuning on LLM-labeled Data May Reduce Reliance on Human Annotation: A Case Study Using Schedule-of-Event Table Detection</a></div>
    <div class="paper-meta">
      📅 2024-08-05
      | 💬 23 pages. Published in MLHC 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated their efficacy across a broad spectrum of tasks in healthcare applications. However, often LLMs need to be fine-tuned on task-specific expert annotated data to achieve optimal performance, which can be expensive and time consuming. In this study, we fine-tune PaLM-2 with parameter efficient fine-tuning (PEFT) using noisy labels obtained from gemini-pro 1.0 for the detection of Schedule-of-Event (SoE) tables, which specify care plan in clinical trial protocols. We introduce a filtering mechanism to select high-confidence labels for this table classification task, thereby reducing the noise in the auto-generated labels. We show that fine-tuned PaLM-2 with those labels achieves performance that exceeds the gemini-pro 1.0 and other LLMs. Furthermore, its performance is close to a PaLM-2 fine-tuned on labels obtained from non-expert annotators. Our results show that leveraging LLM-generated labels through powerful models like gemini-pro can potentially serve as a viable strategy for improving LLM performance through fine-tuning in specialized tasks, particularly in domains where expert annotations are scarce, expensive, or time-consuming to obtain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02559v1">Evaluating and Enhancing LLMs Agent based on Theory of Mind in Guandan: A Multi-Player Cooperative Game under Imperfect Information</a></div>
    <div class="paper-meta">
      📅 2024-08-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown success in handling simple games with imperfect information and enabling multi-agent coordination, but their ability to facilitate practical collaboration against other agents in complex, imperfect information environments, especially in a non-English environment, still needs to be explored. This study investigates the applicability of knowledge acquired by open-source and API-based LLMs to sophisticated text-based games requiring agent collaboration under imperfect information, comparing their performance to established baselines using other types of agents. We propose a Theory of Mind (ToM) planning technique that allows LLM agents to adapt their strategy against various adversaries using only game rules, current state, and historical context as input. An external tool was incorporated to mitigate the challenge of dynamic and extensive action spaces in this card game. Our results show that although a performance gap exists between current LLMs and state-of-the-art reinforcement learning (RL) models, LLMs demonstrate ToM capabilities in this game setting. It consistently improves their performance against opposing agents, suggesting their ability to understand the actions of allies and adversaries and establish collaboration with allies. To encourage further research and understanding, we have made our codebase openly accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02479v1">From LLMs to LLM-based Agents for Software Engineering: A Survey of Current, Challenges and Future</a></div>
    <div class="paper-meta">
      📅 2024-08-05
    </div>
    <details class="paper-abstract">
      With the rise of large language models (LLMs), researchers are increasingly exploring their applications in var ious vertical domains, such as software engineering. LLMs have achieved remarkable success in areas including code generation and vulnerability detection. However, they also exhibit numerous limitations and shortcomings. LLM-based agents, a novel tech nology with the potential for Artificial General Intelligence (AGI), combine LLMs as the core for decision-making and action-taking, addressing some of the inherent limitations of LLMs such as lack of autonomy and self-improvement. Despite numerous studies and surveys exploring the possibility of using LLMs in software engineering, it lacks a clear distinction between LLMs and LLM based agents. It is still in its early stage for a unified standard and benchmarking to qualify an LLM solution as an LLM-based agent in its domain. In this survey, we broadly investigate the current practice and solutions for LLMs and LLM-based agents for software engineering. In particular we summarise six key topics: requirement engineering, code generation, autonomous decision-making, software design, test generation, and software maintenance. We review and differentiate the work of LLMs and LLM-based agents from these six topics, examining their differences and similarities in tasks, benchmarks, and evaluation metrics. Finally, we discuss the models and benchmarks used, providing a comprehensive analysis of their applications and effectiveness in software engineering. We anticipate this work will shed some lights on pushing the boundaries of LLM-based agents in software engineering for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02450v1">An Evaluation of Requirements Modeling for Cyber-Physical Systems via LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-05
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Cyber-physical systems (CPSs) integrate cyber and physical components and enable them to interact with each other to meet user needs. The needs for CPSs span rich application domains such as healthcare and medicine, smart home, smart building, etc. This indicates that CPSs are all about solving real-world problems. With the increasing abundance of sensing devices and effectors, the problems wanted to solve with CPSs are becoming more and more complex. It is also becoming increasingly difficult to extract and express CPS requirements accurately. Problem frame approach aims to shape real-world problems by capturing the characteristics and interconnections of components, where the problem diagram is central to expressing the requirements. CPSs requirements are generally presented in domain-specific documents that are normally expressed in natural language. There is currently no effective way to extract problem diagrams from natural language documents. CPSs requirements extraction and modeling are generally done manually, which is time-consuming, labor-intensive, and error-prone. Large language models (LLMs) have shown excellent performance in natural language understanding. It can be interesting to explore the abilities of LLMs to understand domain-specific documents and identify modeling elements, which this paper is working on. To achieve this goal, we first formulate two tasks (i.e., entity recognition and interaction extraction) and propose a benchmark called CPSBench. Based on this benchmark, extensive experiments are conducted to evaluate the abilities and limitations of seven advanced LLMs. We find some interesting insights. Finally, we establish a taxonomy of LLMs hallucinations in CPSs requirements modeling using problem diagrams. These results will inspire research on the use of LLMs for automated CPSs requirements modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05235v1">SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2024-08-05
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) gain traction, their reliance on power-hungry GPUs places ever-increasing energy demands, raising environmental and monetary concerns. Inference dominates LLM workloads, presenting a critical challenge for providers: minimizing energy costs under Service-Level Objectives (SLOs) that ensure optimal user experience. In this paper, we present \textit{throttLL'eM}, a framework that reduces energy consumption while meeting SLOs through the use of instance and GPU frequency scaling. \textit{throttLL'eM} features mechanisms that project future KV cache usage and batch size. Leveraging a Machine-Learning (ML) model that receives these projections as inputs, \textit{throttLL'eM} manages performance at the iteration level to satisfy SLOs with reduced frequencies and instance sizes. We show that the proposed ML model achieves $R^2$ scores greater than 0.97 and miss-predicts performance by less than 1 iteration per second on average. Experimental results on LLM inference traces show that \textit{throttLL'eM} achieves up to 43.8\% lower energy consumption and an energy efficiency improvement of at least $1.71\times$ under SLOs, when compared to NVIDIA's Triton server.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01112v2">Agentic LLM Workflows for Generating Patient-Friendly Medical Reports</a></div>
    <div class="paper-meta">
      📅 2024-08-05
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The application of Large Language Models (LLMs) in healthcare is expanding rapidly, with one potential use case being the translation of formal medical reports into patient-legible equivalents. Currently, LLM outputs often need to be edited and evaluated by a human to ensure both factual accuracy and comprehensibility, and this is true for the above use case. We aim to minimize this step by proposing an agentic workflow with the Reflexion framework, which uses iterative self-reflection to correct outputs from an LLM. This pipeline was tested and compared to zero-shot prompting on 16 randomized radiology reports. In our multi-agent approach, reports had an accuracy rate of 94.94% when looking at verification of ICD-10 codes, compared to zero-shot prompted reports, which had an accuracy rate of 68.23%. Additionally, 81.25% of the final reflected reports required no corrections for accuracy or readability, while only 25% of zero-shot prompted reports met these criteria without needing modifications. These results indicate that our approach presents a feasible method for communicating clinical findings to patients in a quick, efficient and coherent manner whilst also retaining medical accuracy. The codebase is available for viewing at http://github.com/malavikhasudarshan/Multi-Agent-Patient-Letter-Generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02193v1">CodeACT: Code Adaptive Compute-efficient Tuning Framework for Code LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great potential in code-related tasks, yet open-source models lag behind their closed-source counterparts. To bridge this performance gap, existing methods generate vast amounts of synthetic data for fine-tuning, leading to inefficiencies in training. Motivated by the need for more effective and efficient training, we propose the Code Adaptive Compute-efficient Tuning (CodeACT) framework. CodeACT introduces the Complexity and Diversity Aware Sampling (CDAS) method to select high-quality training data based on complexity and diversity, and the Dynamic Pack padding strategy to reduce computational resource usage by minimizing padding tokens during training. Experimental results demonstrate that CodeACT-DeepSeek-Coder-6.7B, fine-tuned on only 40% of the EVOL-Instruct data, achieves an 8.6% performance increase on HumanEval, reduces training time by 78%, and decreases peak GPU memory usage by 27%. These findings underscore CodeACT's ability to enhance the performance and efficiency of open-source models. By optimizing both the data selection and training processes, CodeACT offers a comprehensive approach to improving the capabilities of open-source LLMs while significantly reducing computational requirements, addressing the dual challenges of data quality and training efficiency, and paving the way for more resource-efficient and performant models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11058v1">LLM Agents Improve Semantic Code Search</a></div>
    <div class="paper-meta">
      📅 2024-08-05
      | 💬 12 pages, 1 Figure
    </div>
    <details class="paper-abstract">
      Code Search is a key task that many programmers often have to perform while developing solutions to problems. Current methodologies suffer from an inability to perform accurately on prompts that contain some ambiguity or ones that require additional context relative to a code-base. We introduce the approach of using Retrieval Augmented Generation (RAG) powered agents to inject information into user prompts allowing for better inputs into embedding models. By utilizing RAG, agents enhance user queries with relevant details from GitHub repositories, making them more informative and contextually aligned. Additionally, we introduce a multi-stream ensemble approach which when paired with agentic workflow can obtain improved retrieval accuracy, which we deploy on application called repo-rift.com. Experimental results on the CodeSearchNet dataset demonstrate that RepoRift significantly outperforms existing methods, achieving an 78.2% success rate at Success@10 and a 34.6% success rate at Success@1. This research presents a substantial advancement in semantic code search, highlighting the potential of agentic LLMs and RAG to enhance code retrieval systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02143v1">Analyzing Cultural Representations of Emotions in LLMs through Mixed Emotion Survey</a></div>
    <div class="paper-meta">
      📅 2024-08-04
      | 💬 Was accepted to ACII 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained widespread global adoption, showcasing advanced linguistic capabilities across multiple of languages. There is a growing interest in academia to use these models to simulate and study human behaviors. However, it is crucial to acknowledge that an LLM's proficiency in a specific language might not fully encapsulate the norms and values associated with its culture. Concerns have emerged regarding potential biases towards Anglo-centric cultures and values due to the predominance of Western and US-based training data. This study focuses on analyzing the cultural representations of emotions in LLMs, in the specific case of mixed-emotion situations. Our methodology is based on the studies of Miyamoto et al. (2010), which identified distinctive emotional indicators in Japanese and American human responses. We first administer their mixed emotion survey to five different LLMs and analyze their outputs. Second, we experiment with contextual variables to explore variations in responses considering both language and speaker origin. Thirdly, we expand our investigation to encompass additional East Asian and Western European origin languages to gauge their alignment with their respective cultures, anticipating a closer fit. We find that (1) models have limited alignment with the evidence in the literature; (2) written language has greater effect on LLMs' response than information on participants origin; and (3) LLMs responses were found more similar for East Asian languages than Western European languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01469v3">LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples</a></div>
    <div class="paper-meta">
      📅 2024-08-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), including GPT-3.5, LLaMA, and PaLM, seem to be knowledgeable and able to adapt to many tasks. However, we still cannot completely trust their answers, since LLMs suffer from \textbf{hallucination}\textemdash fabricating non-existent facts, deceiving users with or without their awareness. However, the reasons for their existence and pervasiveness remain unclear. In this paper, we demonstrate that nonsensical prompts composed of random tokens can also elicit the LLMs to respond with hallucinations. Moreover, we provide both theoretical and experimental evidence that transformers can be manipulated to produce specific pre-define tokens by perturbing its input sequence. This phenomenon forces us to revisit that \emph{hallucination may be another view of adversarial examples}, and it shares similar characteristics with conventional adversarial examples as a basic property of LLMs. Therefore, we formalize an automatic hallucination triggering method as the \textit{hallucination attack} in an adversarial way. Finally, we explore the basic properties of attacked adversarial prompts and propose a simple yet effective defense strategy. Our code is released on GitHub\footnote{https://github.com/PKU-YuanGroup/Hallucination-Attack}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01993v1">Towards Automatic Hands-on-Keyboard Attack Detection Using LLMs in EDR Solutions</a></div>
    <div class="paper-meta">
      📅 2024-08-04
    </div>
    <details class="paper-abstract">
      Endpoint Detection and Remediation (EDR) platforms are essential for identifying and responding to cyber threats. This study presents a novel approach using Large Language Models (LLMs) to detect Hands-on-Keyboard (HOK) cyberattacks. Our method involves converting endpoint activity data into narrative forms that LLMs can analyze to distinguish between normal operations and potential HOK attacks. We address the challenges of interpreting endpoint data by segmenting narratives into windows and employing a dual training strategy. The results demonstrate that LLM-based models have the potential to outperform traditional machine learning methods, offering a promising direction for enhancing EDR capabilities and apply LLMs in cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01869v1">MALADE: Orchestration of LLM-powered Agents with Retrieval Augmented Generation for Pharmacovigilance</a></div>
    <div class="paper-meta">
      📅 2024-08-03
      | 💬 Paper published at Machine Learning for Healthcare 2024 (MLHC'24)
    </div>
    <details class="paper-abstract">
      In the era of Large Language Models (LLMs), given their remarkable text understanding and generation abilities, there is an unprecedented opportunity to develop new, LLM-based methods for trustworthy medical knowledge synthesis, extraction and summarization. This paper focuses on the problem of Pharmacovigilance (PhV), where the significance and challenges lie in identifying Adverse Drug Events (ADEs) from diverse text sources, such as medical literature, clinical notes, and drug labels. Unfortunately, this task is hindered by factors including variations in the terminologies of drugs and outcomes, and ADE descriptions often being buried in large amounts of narrative text. We present MALADE, the first effective collaborative multi-agent system powered by LLM with Retrieval Augmented Generation for ADE extraction from drug label data. This technique involves augmenting a query to an LLM with relevant information extracted from text resources, and instructing the LLM to compose a response consistent with the augmented data. MALADE is a general LLM-agnostic architecture, and its unique capabilities are: (1) leveraging a variety of external sources, such as medical literature, drug labels, and FDA tools (e.g., OpenFDA drug information API), (2) extracting drug-outcome association in a structured format along with the strength of the association, and (3) providing explanations for established associations. Instantiated with GPT-4 Turbo or GPT-4o, and FDA drug label data, MALADE demonstrates its efficacy with an Area Under ROC Curve of 0.90 against the OMOP Ground Truth table of ADEs. Our implementation leverages the Langroid multi-agent LLM framework and can be found at https://github.com/jihyechoi77/malade.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01867v1">TrustNavGPT: Modeling Uncertainty to Improve Trustworthiness of Audio-Guided LLM-Based Robot Navigation</a></div>
    <div class="paper-meta">
      📅 2024-08-03
    </div>
    <details class="paper-abstract">
      While LLMs are proficient at processing text in human conversations, they often encounter difficulties with the nuances of verbal instructions and, thus, remain prone to hallucinate trust in human command. In this work, we present TrustNavGPT, an LLM based audio guided navigation agent that uses affective cues in spoken communication elements such as tone and inflection that convey meaning beyond words, allowing it to assess the trustworthiness of human commands and make effective, safe decisions. Our approach provides a lightweight yet effective approach that extends existing LLMs to model audio vocal features embedded in the voice command and model uncertainty for safe robotic navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04650v1">Building Trust in Mental Health Chatbots: Safety Metrics and LLM-Based Evaluation Tools</a></div>
    <div class="paper-meta">
      📅 2024-08-03
    </div>
    <details class="paper-abstract">
      Objective: This study aims to develop and validate an evaluation framework to ensure the safety and reliability of mental health chatbots, which are increasingly popular due to their accessibility, human-like interactions, and context-aware support. Materials and Methods: We created an evaluation framework with 100 benchmark questions and ideal responses, and five guideline questions for chatbot responses. This framework, validated by mental health experts, was tested on a GPT-3.5-turbo-based chatbot. Automated evaluation methods explored included large language model (LLM)-based scoring, an agentic approach using real-time data, and embedding models to compare chatbot responses against ground truth standards. Results: The results highlight the importance of guidelines and ground truth for improving LLM evaluation accuracy. The agentic method, dynamically accessing reliable information, demonstrated the best alignment with human assessments. Adherence to a standardized, expert-validated framework significantly enhanced chatbot response safety and reliability. Discussion: Our findings emphasize the need for comprehensive, expert-tailored safety evaluation metrics for mental health chatbots. While LLMs have significant potential, careful implementation is necessary to mitigate risks. The superior performance of the agentic approach underscores the importance of real-time data access in enhancing chatbot reliability. Conclusion: The study validated an evaluation framework for mental health chatbots, proving its effectiveness in improving safety and reliability. Future work should extend evaluations to accuracy, bias, empathy, and privacy to ensure holistic assessment and responsible integration into healthcare. Standardized evaluations will build trust among users and professionals, facilitating broader adoption and improved mental health support through technology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06399v3">Should We Fine-Tune or RAG? Evaluating Different Techniques to Adapt LLMs for Dialogue</a></div>
    <div class="paper-meta">
      📅 2024-08-03
      | 💬 Accepted at INLG 2024
    </div>
    <details class="paper-abstract">
      We study the limitations of Large Language Models (LLMs) for the task of response generation in human-machine dialogue. Several techniques have been proposed in the literature for different dialogue types (e.g., Open-Domain). However, the evaluations of these techniques have been limited in terms of base LLMs, dialogue types and evaluation metrics. In this work, we extensively analyze different LLM adaptation techniques when applied to different dialogue types. We have selected two base LLMs, Llama-2 and Mistral, and four dialogue types Open-Domain, Knowledge-Grounded, Task-Oriented, and Question Answering. We evaluate the performance of in-context learning and fine-tuning techniques across datasets selected for each dialogue type. We assess the impact of incorporating external knowledge to ground the generation in both scenarios of Retrieval-Augmented Generation (RAG) and gold knowledge. We adopt consistent evaluation and explainability criteria for automatic metrics and human evaluation protocols. Our analysis shows that there is no universal best-technique for adapting large language models as the efficacy of each technique depends on both the base LLM and the specific type of dialogue. Last but not least, the assessment of the best adaptation technique should include human evaluation to avoid false expectations and outcomes derived from automatic metrics.
    </details>
</div>
