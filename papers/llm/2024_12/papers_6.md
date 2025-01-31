# llm - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14279v3">ZS4C: Zero-Shot Synthesis of Compilable Code for Incomplete Code Snippets using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 This paper has been accepted and published in ACM Transactions on Software Engineering and Methodology (TOSEM), [2024], [https://dl.acm.org/doi/10.1145/3702979]
    </div>
    <details class="paper-abstract">
      Technical Q&A sites are valuable for software developers seeking knowledge, but the code snippets they provide are often uncompilable and incomplete due to unresolved types and missing libraries. This poses a challenge for users who wish to reuse or analyze these snippets. Existing methods either do not focus on creating compilable code or have low success rates. To address this, we propose ZS4C, a lightweight approach for zero-shot synthesis of compilable code from incomplete snippets using Large Language Models (LLMs). ZS4C operates in two stages: first, it uses an LLM, like GPT-3.5, to identify missing import statements in a snippet; second, it collaborates with a validator (e.g., compiler) to fix compilation errors caused by incorrect imports and syntax issues. We evaluated ZS4C on the StatType-SO benchmark and a new dataset, Python-SO, which includes 539 Python snippets from Stack Overflow across the 20 most popular Python libraries. ZS4C significantly outperforms existing methods, improving the compilation rate from 63% to 95.1% compared to the state-of-the-art SnR, marking a 50.1% improvement. On average, ZS4C can infer more accurate import statements (with an F1 score of 0.98) than SnR, with an improvement of 8.5% in the F1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06738v1">JAPAGEN: Efficient Few/Zero-shot Learning via Japanese Training Dataset Generation with LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 Accepted by PACLIC38 (2024)
    </div>
    <details class="paper-abstract">
      Recently some studies have highlighted the potential of Large Language Models (LLMs) as effective generators of supervised training data, offering advantages such as enhanced inference efficiency and reduced costs associated with data collection. However, these studies have predominantly focused on English language tasks. In this paper, we address the fundamental research question: Can LLMs serve as proficient training data generators for other language tasks? Specifically, we leverage LLMs to synthesize supervised training data under few-shot and zero-shot learning scenarios across six diverse Japanese downstream tasks. Subsequently, we utilize this synthesized data to train compact models (e.g., BERT). This novel methodology is termed JAPAGEN. Our experimental findings underscore that JAPAGEN achieves robust performance in classification tasks that necessitate formal text inputs, demonstrating competitive results compared to conventional LLM prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06681v1">Toward LLM-Agent-Based Modeling of Transportation Systems: A Conceptual Framework</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      In transportation system demand modeling and simulation, agent-based models and microsimulations are current state-of-the-art approaches. However, existing agent-based models still have some limitations on behavioral realism and resource demand that limit their applicability. In this study, leveraging the emerging technology of large language models (LLMs) and LLM-based agents, we propose a general LLM-agent-based modeling framework for transportation systems. We argue that LLM agents not only possess the essential capabilities to function as agents but also offer promising solutions to overcome some limitations of existing agent-based models. Our conceptual framework design closely replicates the decision-making and interaction processes and traits of human travelers within transportation networks, and we demonstrate that the proposed systems can meet critical behavioral criteria for decision-making and learning behaviors using related studies and a demonstrative example of LLM agents' learning and adjustment in the bottleneck setting. Although further refinement of the LLM-agent-based modeling framework is necessary, we believe that this approach has the potential to improve transportation system modeling and simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16176v1">Efficient VoIP Communications through LLM-based Real-Time Speech Reconstruction and Call Prioritization for Emergency Services</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 15 pages,8 figures
    </div>
    <details class="paper-abstract">
      Emergency communication systems face disruptions due to packet loss, bandwidth constraints, poor signal quality, delays, and jitter in VoIP systems, leading to degraded real-time service quality. Victims in distress often struggle to convey critical information due to panic, speech disorders, and background noise, further complicating dispatchers' ability to assess situations accurately. Staffing shortages in emergency centers exacerbate delays in coordination and assistance. This paper proposes leveraging Large Language Models (LLMs) to address these challenges by reconstructing incomplete speech, filling contextual gaps, and prioritizing calls based on severity. The system integrates real-time transcription with Retrieval-Augmented Generation (RAG) to generate contextual responses, using Twilio and AssemblyAI APIs for seamless implementation. Evaluation shows high precision, favorable BLEU and ROUGE scores, and alignment with real-world needs, demonstrating the model's potential to optimize emergency response workflows and prioritize critical cases effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16851v2">Can tweets predict article retractions? A comparison between human and LLM labelling</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Quickly detecting problematic research articles is crucial to safeguarding the integrity of scientific research. This study explores whether Twitter mentions of retracted articles can signal potential problems with the articles prior to their retraction, potentially serving as an early warning system for scholars. To investigate this, we analysed a dataset of 4,354 Twitter mentions associated with 504 retracted articles. The effectiveness of Twitter mentions in predicting article retractions was evaluated by both manual and Large Language Model (LLM) labelling. Manual labelling results indicated that 25.7% of tweets signalled problems before retraction. Using the manual labelling results as the baseline, we found that LLMs (GPT-4o-mini, Gemini 1.5 Flash, and Claude-3.5-Haiku) outperformed lexicon-based sentiment analysis tools (e.g., TextBlob) in detecting potential problems, suggesting that automatic detection of problematic articles from social media using LLMs is technically feasible. Nevertheless, since only a small proportion of retracted articles (11.1%) were criticised on Twitter prior to retraction, such automatic systems would detect only a minority of problematic articles. Overall, this study offers insights into how social media data, coupled with emerging generative AI techniques, can support research integrity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06432v1">Integrating Expert Labels into LLM-based Emission Goal Detection: Example Selection vs Automatic Prompt Design</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      We address the detection of emission reduction goals in corporate reports, an important task for monitoring companies' progress in addressing climate change. Specifically, we focus on the issue of integrating expert feedback in the form of labeled example passages into LLM-based pipelines, and compare the two strategies of (1) a dynamic selection of few-shot examples and (2) the automatic optimization of the prompt by the LLM itself. Our findings on a public dataset of 769 climate-related passages from real-world business reports indicate that automatic prompt optimization is the superior approach, while combining both methods provides only limited benefit. Qualitative results indicate that optimized prompts do indeed capture many intricacies of the targeted emission goal extraction task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06370v1">Exploring Memorization and Copyright Violation in Frontier LLMs: A Study of the New York Times v. OpenAI 2023 Lawsuit</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      Copyright infringement in frontier LLMs has received much attention recently due to the New York Times v. OpenAI lawsuit, filed in December 2023. The New York Times claims that GPT-4 has infringed its copyrights by reproducing articles for use in LLM training and by memorizing the inputs, thereby publicly displaying them in LLM outputs. Our work aims to measure the propensity of OpenAI's LLMs to exhibit verbatim memorization in its outputs relative to other LLMs, specifically focusing on news articles. We discover that both GPT and Claude models use refusal training and output filters to prevent verbatim output of the memorized articles. We apply a basic prompt template to bypass the refusal training and show that OpenAI models are currently less prone to memorization elicitation than models from Meta, Mistral, and Anthropic. We find that as models increase in size, especially beyond 100 billion parameters, they demonstrate significantly greater capacity for memorization. Our findings have practical implications for training: more attention must be placed on preventing verbatim memorization in very large models. Our findings also have legal significance: in assessing the relative memorization capacity of OpenAI's LLMs, we probe the strength of The New York Times's copyright infringement claims and OpenAI's legal defenses, while underscoring issues at the intersection of generative AI, law, and policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.08339v5">Assessing the potential of LLM-assisted annotation for corpus-based pragmatics and discourse analysis: The case of apology</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 24 pages, 2 figures, 3 tablels
    </div>
    <details class="paper-abstract">
      Certain forms of linguistic annotation, like part of speech and semantic tagging, can be automated with high accuracy. However, manual annotation is still necessary for complex pragmatic and discursive features that lack a direct mapping to lexical forms. This manual process is time-consuming and error-prone, limiting the scalability of function-to-form approaches in corpus linguistics. To address this, our study explores the possibility of using large language models (LLMs) to automate pragma-discursive corpus annotation. We compare GPT-3.5 (the model behind the free-to-use version of ChatGPT), GPT-4 (the model underpinning the precise mode of Bing chatbot), and a human coder in annotating apology components in English based on the local grammar framework. We find that GPT-4 outperformed GPT-3.5, with accuracy approaching that of a human coder. These results suggest that LLMs can be successfully deployed to aid pragma-discursive corpus annotation, making the process more efficient, scalable and accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01432v3">Split and Merge: Aligning Position Biases in LLM-based Evaluators</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 Accepted by EMNLP 2024. Please cite the conference version of this paper, e.g., "Zongjie Li, Chaozheng Wang, Pingchuan Ma, Daoyuan Wu, Shuai Wang, Cuiyun Gao, and Yang Liu. 2024. Split and Merge: Aligning Position Biases in LLM-based Evaluators. (EMNLP 2024)"
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise as automated evaluators for assessing the quality of answers generated by AI systems. However, these LLM-based evaluators exhibit position bias, or inconsistency, when used to evaluate candidate answers in pairwise comparisons, favoring either the first or second answer regardless of content. To address this limitation, we propose PORTIA, an alignment-based system designed to mimic human comparison strategies to calibrate position bias in a lightweight yet effective manner. Specifically, PORTIA splits the answers into multiple segments, aligns similar content across candidate answers, and then merges them back into a single prompt for evaluation by LLMs. We conducted extensive experiments with six diverse LLMs to evaluate 11,520 answer pairs. Our results show that PORTIA markedly enhances the consistency rates for all the models and comparison forms tested, achieving an average relative improvement of 47.46%. Remarkably, PORTIA enables less advanced GPT models to achieve 88% agreement with the state-of-the-art GPT-4 model at just 10% of the cost. Furthermore, it rectifies around 80% of the position bias instances within the GPT-4 model, elevating its consistency rate up to 98%. Subsequent human evaluations indicate that the PORTIA-enhanced GPT-3.5 model can even surpass the standalone GPT-4 in terms of alignment with human evaluators. These findings highlight PORTIA's ability to correct position bias, improve LLM consistency, and boost performance while keeping cost-efficiency. This represents a valuable step toward a more reliable and scalable use of LLMs for automated evaluations across diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03817v3">From Novice to Expert: LLM Agent Policy Optimization via Step-wise Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      The outstanding capabilities of large language models (LLMs) render them a crucial component in various autonomous agent systems. While traditional methods depend on the inherent knowledge of LLMs without fine-tuning, more recent approaches have shifted toward the reinforcement learning strategy to further enhance agents' ability to solve complex interactive tasks with environments and tools. However, previous approaches are constrained by the sparse reward issue, where existing datasets solely provide a final scalar reward for each multi-step reasoning chain, potentially leading to ineffectiveness and inefficiency in policy learning. In this paper, we introduce StepAgent, which utilizes step-wise reward to optimize the agent's reinforcement learning process. Inheriting the spirit of novice-to-expert theory, we first compare the actions of the expert and the agent to automatically generate intermediate rewards for fine-grained optimization. Additionally, we propose implicit-reward and inverse reinforcement learning techniques to facilitate agent reflection and policy adjustment. Further theoretical analysis demonstrates that the action distribution of the agent can converge toward the expert action distribution over multiple training cycles. Experimental results across various datasets indicate that StepAgent outperforms existing baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06294v1">Beyond pip install: Evaluating LLM Agents for the Automated Installation of Python Projects</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      Many works have recently proposed the use of Large Language Model (LLM) based agents for performing `repository level' tasks, loosely defined as a set of tasks whose scopes are greater than a single file. This has led to speculation that the orchestration of these repository-level tasks could lead to software engineering agents capable of performing almost independently of human intervention. However, of the suite of tasks that would need to be performed by this autonomous software engineering agent, we argue that one important task is missing, which is to fulfil project level dependency by installing other repositories. To investigate the feasibility of this repository level installation task, we introduce a benchmark of of repository installation tasks curated from 40 open source Python projects, which includes a ground truth installation process for each target repository. Further, we propose Installamatic, an agent which aims to perform and verify the installation of a given repository by searching for relevant instructions from documentation in the repository. Empirical experiments reveal that that 55% of the studied repositories can be automatically installed by our agent at least one out of ten times. Through further analysis, we identify the common causes for our agent's inability to install a repository, discuss the challenges faced in the design and implementation of such an agent and consider the implications that such an agent could have for developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00627v2">ARChef: An iOS-Based Augmented Reality Cooking Assistant Powered by Multimodal Gemini LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      Cooking meals can be difficult, causing many to resort to cookbooks and online recipes. However, relying on these traditional methods of cooking often results in missing ingredients, nutritional hazards, and unsatisfactory meals. Using Augmented Reality (AR) can address these issues; however, current AR cooking applications have poor user interfaces and limited accessibility. This paper proposes a prototype of an iOS application that integrates AR and Computer Vision (CV) into the cooking process. We leverage Google's Gemini Large Language Model (LLM) to identify ingredients in the camera's field of vision and generate recipe choices with detailed nutritional information. Additionally, this application uses Apple's ARKit to create an AR user interface compatible with iOS devices. Users can personalize their meal suggestions by inputting their dietary preferences and rating each meal. The application's effectiveness is evaluated through three rounds of user experience surveys. This application advances the field of accessible cooking assistance technologies, aiming to reduce food wastage and improve the meal planning experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06272v1">Methods for Legal Citation Prediction in the Age of LLMs: An Australian Law Case Study</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 For code, data, and models see https://auslawbench.github.io
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have shown great potential across a wide range of legal tasks. Despite these advances, mitigating hallucination remains a significant challenge, with state-of-the-art LLMs still frequently generating incorrect legal references. In this paper, we focus on the problem of legal citation prediction within the Australian law context, where correctly identifying and citing relevant legislations or precedents is critical. We compare several approaches: prompting general purpose and law-specialised LLMs, retrieval-only pipelines with both generic and domain-specific embeddings, task-specific instruction-tuning of LLMs, and hybrid strategies that combine LLMs with retrieval augmentation, query expansion, or voting ensembles. Our findings indicate that domain-specific pre-training alone is insufficient for achieving satisfactory citation accuracy even after law-specialised pre-training. In contrast, instruction tuning on our task-specific dataset dramatically boosts performance reaching the best results across all settings. We also highlight that database granularity along with the type of embeddings play a critical role in the performance of retrieval systems. Among retrieval-based approaches, hybrid methods consistently outperform retrieval-only setups, and among these, ensemble voting delivers the best result by combining the predictive quality of instruction-tuned LLMs with the retrieval system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10415v1">Generative Adversarial Reviews: When LLMs Become the Critic</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      The peer review process is fundamental to scientific progress, determining which papers meet the quality standards for publication. Yet, the rapid growth of scholarly production and increasing specialization in knowledge areas strain traditional scientific feedback mechanisms. In light of this, we introduce Generative Agent Reviewers (GAR), leveraging LLM-empowered agents to simulate faithful peer reviewers. To enable generative reviewers, we design an architecture that extends a large language model with memory capabilities and equips agents with reviewer personas derived from historical data. Central to this approach is a graph-based representation of manuscripts, condensing content and logically organizing information - linking ideas with evidence and technical details. GAR's review process leverages external knowledge to evaluate paper novelty, followed by detailed assessment using the graph representation and multi-round assessment. Finally, a meta-reviewer aggregates individual reviews to predict the acceptance decision. Our experiments demonstrate that GAR performs comparably to human reviewers in providing detailed feedback and predicting paper outcomes. Beyond mere performance comparison, we conduct insightful experiments, such as evaluating the impact of reviewer expertise and examining fairness in reviews. By offering early expert-level feedback, typically restricted to a limited group of researchers, GAR democratizes access to transparent and in-depth evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06229v1">LLMs as Debate Partners: Utilizing Genetic Algorithms and Adversarial Search for Adaptive Arguments</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      This paper introduces DebateBrawl, an innovative AI-powered debate platform that integrates Large Language Models (LLMs), Genetic Algorithms (GA), and Adversarial Search (AS) to create an adaptive and engaging debating experience. DebateBrawl addresses the limitations of traditional LLMs in strategic planning by incorporating evolutionary optimization and game-theoretic techniques. The system demonstrates remarkable performance in generating coherent, contextually relevant arguments while adapting its strategy in real-time. Experimental results involving 23 debates show balanced outcomes between AI and human participants, with the AI system achieving an average score of 2.72 compared to the human average of 2.67 out of 10. User feedback indicates significant improvements in debating skills and a highly satisfactory learning experience, with 85% of users reporting improved debating abilities and 78% finding the AI opponent appropriately challenging. The system's ability to maintain high factual accuracy (92% compared to 78% in human-only debates) while generating diverse arguments addresses critical concerns in AI-assisted discourse. DebateBrawl not only serves as an effective educational tool but also contributes to the broader goal of improving public discourse through AI-assisted argumentation. The paper discusses the ethical implications of AI in persuasive contexts and outlines the measures implemented to ensure responsible development and deployment of the system, including robust fact-checking mechanisms and transparency in decision-making processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10053v2">Householder Pseudo-Rotation: A Novel Approach to Activation Editing in LLMs with Direction-Magnitude Perspective</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      Activation Editing, which involves directly editting the internal representations of large language models (LLMs) to alter their behaviors and achieve desired properties, has emerged as a promising area of research. Existing works primarily treat LLMs' activations as points in space and modify them by adding steering vectors. However, this approach is limited in its ability to achieve greater performance improvement while maintaining the necessary consistency of activation magnitudes. To overcome these issues, we propose a novel editing method that views activations in terms of their directions and magnitudes. Our method, named Householder Pseudo-Rotation (HPR), mimics the rotation transformation, thus preserving activation norms and resulting in an improved performance on various safety benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06181v1">Enhancing Adversarial Resistance in LLMs with Recursion</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      The increasing integration of Large Language Models (LLMs) into society necessitates robust defenses against vulnerabilities from jailbreaking and adversarial prompts. This project proposes a recursive framework for enhancing the resistance of LLMs to manipulation through the use of prompt simplification techniques. By increasing the transparency of complex and confusing adversarial prompts, the proposed method enables more reliable detection and prevention of malicious inputs. Our findings attempt to address a critical problem in AI safety and security, providing a foundation for the development of systems able to distinguish harmless inputs from prompts containing malicious intent. As LLMs continue to be used in diverse applications, the importance of such safeguards will only grow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14733v1">LLM as HPC Expert: Extending RAG Architecture for HPC Data</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      High-Performance Computing (HPC) is crucial for performing advanced computational tasks, yet their complexity often challenges users, particularly those unfamiliar with HPC-specific commands and workflows. This paper introduces Hypothetical Command Embeddings (HyCE), a novel method that extends Retrieval-Augmented Generation (RAG) by integrating real-time, user-specific HPC data, enhancing accessibility to these systems. HyCE enriches large language models (LLM) with real-time, user-specific HPC information, addressing the limitations of fine-tuned models on such data. We evaluate HyCE using an automated RAG evaluation framework, where the LLM itself creates synthetic questions from the HPC data and serves as a judge, assessing the efficacy of the extended RAG with the evaluation metrics relevant for HPC tasks. Additionally, we tackle essential security concerns, including data privacy and command execution risks, associated with deploying LLMs in HPC environments. This solution provides a scalable and adaptable approach for HPC clusters to leverage LLMs as HPC expert, bridging the gap between users and the complex systems of HPC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06860v1">Balancing Efficiency and Effectiveness: An LLM-Infused Approach for Optimized CTR Prediction</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 5 pages, 4 figures,4 tables
    </div>
    <details class="paper-abstract">
      Click-Through Rate (CTR) prediction is essential in online advertising, where semantic information plays a pivotal role in shaping user decisions and enhancing CTR effectiveness. Capturing and modeling deep semantic information, such as a user's preference for "H\"aagen-Dazs' HEAVEN strawberry light ice cream" due to its health-conscious and premium attributes, is challenging. Traditional semantic modeling often overlooks these intricate details at the user and item levels. To bridge this gap, we introduce a novel approach that models deep semantic information end-to-end, leveraging the comprehensive world knowledge capabilities of Large Language Models (LLMs). Our proposed LLM-infused CTR prediction framework(Multi-level Deep Semantic Information Infused CTR model via Distillation, MSD) is designed to uncover deep semantic insights by utilizing LLMs to extract and distill critical information into a smaller, more efficient model, enabling seamless end-to-end training and inference. Importantly, our framework is carefully designed to balance efficiency and effectiveness, ensuring that the model not only achieves high performance but also operates with optimal resource utilization. Online A/B tests conducted on the Meituan sponsored-search system demonstrate that our method significantly outperforms baseline models in terms of Cost Per Mile (CPM) and CTR, validating its effectiveness, scalability, and balanced approach in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10414v1">Exploring Complex Mental Health Symptoms via Classifying Social Media Data with Explainable LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-09
      | 💬 Accepted to Machine Learning for Health (ML4H) Findings 2024 (co-located with NeurIPS 2024)
    </div>
    <details class="paper-abstract">
      We propose a pipeline for gaining insights into complex diseases by training LLMs on challenging social media text data classification tasks, obtaining explanations for the classification outputs, and performing qualitative and quantitative analysis on the explanations. We report initial results on predicting, explaining, and systematizing the explanations of predicted reports on mental health concerns in people reporting Lyme disease concerns. We report initial results on predicting future ADHD concerns for people reporting anxiety disorder concerns, and demonstrate preliminary results on visualizing the explanations for predicting that a person with anxiety concerns will in the future have ADHD concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05276v3">GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching</a></div>
    <div class="paper-meta">
      📅 2024-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as GPT, have revolutionized artificial intelligence by enabling nuanced understanding and generation of human-like text across a wide range of applications. However, the high computational and financial costs associated with frequent API calls to these models present a substantial bottleneck, especially for applications like customer service chatbots that handle repetitive queries. In this paper, we introduce GPT Semantic Cache, a method that leverages semantic caching of query embeddings in in-memory storage (Redis). By storing embeddings of user queries, our approach efficiently identifies semantically similar questions, allowing for the retrieval of pre-generated responses without redundant API calls to the LLM. This technique achieves a notable reduction in operational costs while significantly enhancing response times, making it a robust solution for optimizing LLM-powered applications. Our experiments demonstrate that GPT Semantic Cache reduces API calls by up to 68.8% across various query categories, with cache hit rates ranging from 61.6% to 68.8%. Additionally, the system achieves high accuracy, with positive hit rates exceeding 97%, confirming the reliability of cached responses. This technique not only reduces operational costs, but also improves response times, enhancing the efficiency of LLM-powered applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06858v1">Taming Sensitive Weights : Noise Perturbation Fine-tuning for Robust LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2024-12-08
      | 💬 Submitted to CPAL 2024
    </div>
    <details class="paper-abstract">
      Quantization is a critical step to enable efficient LLM serving under limited resource. However, previous research observes that certain weights in the LLM, known as outliers, are significantly sensitive to quantization noises. Existing quantization methods leave these outliers as floating points or higher precisions to retain performance, posting challenges on the efficient hardware deployment of the mixed-precision model. This work investigates an alternative way to tame the sensitive weights' impact on the quantization error, by reducing the loss Hessian trace with respect to outliers through an efficient fine-tuning process. We propose Noise Perturbation Fine-tuning (NPFT), which identifies outlier weights and add random weight perturbations on the outliers as the model going through a PEFT optimization. NPFT tames the sensitivity of outlier weights so that the quantized model performance can be improved without special treatment to the outliers. When applied to OPT and LLaMA models, our NPFT method achieves stable performance improvements for both uniform and non-uniform quantizers, while also offering better inference efficiency. Notably, the simplest RTN can achieve performance on par with GPTQ using our NPFT on LLaMA2-7B-4bits benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19146v3">Puzzle: Distillation-Based NAS for Inference-Optimized LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities, but their adoption is limited by high computational costs during inference. While increasing parameter counts enhances accuracy, it also widens the gap between state-of-the-art capabilities and practical deployability. We present Puzzle, a framework to accelerate LLM inference on specific hardware while preserving their capabilities. Through an innovative application of neural architecture search (NAS) at an unprecedented scale, Puzzle systematically optimizes models with tens of billions of parameters under hardware constraints. Our approach utilizes blockwise local knowledge distillation (BLD) for parallel architecture exploration and employs mixed-integer programming for precise constraint optimization. We demonstrate the real-world impact of our framework through Llama-3.1-Nemotron-51B-Instruct (Nemotron-51B), a publicly available model derived from Llama-3.1-70B-Instruct. Nemotron-51B achieves a 2.17x inference throughput speedup, fitting on a single NVIDIA H100 GPU while preserving 98.4% of the original model's capabilities. Nemotron-51B currently stands as the most accurate language model capable of inference on a single GPU with large batch sizes. Remarkably, this transformation required just 45B training tokens, compared to over 15T tokens used for the 70B model it was derived from. This establishes a new paradigm where powerful models can be optimized for efficient deployment with only negligible compromise of their capabilities, demonstrating that inference performance, not parameter count alone, should guide model selection. With the release of Nemotron-51B and the presentation of the Puzzle framework, we provide practitioners immediate access to state-of-the-art language modeling capabilities at significantly reduced computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03096v2">TOOL-ED: Enhancing Empathetic Response Generation with the Tool Calling Capability of LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-08
    </div>
    <details class="paper-abstract">
      Empathetic conversation is a crucial characteristic in daily conversations between individuals. Nowadays, Large Language models (LLMs) have shown outstanding performance in generating empathetic responses. Knowledge bases like COMET can assist LLMs in mitigating illusions and enhancing the understanding of users' intentions and emotions. However, models remain heavily reliant on fixed knowledge bases and unrestricted incorporation of external knowledge can introduce noise. Tool learning is a flexible end-to-end approach that assists LLMs in handling complex problems. In this paper, we propose Emotional Knowledge Tool Calling (EKTC) framework, which encapsulates the commonsense knowledge bases as empathetic tools, enabling LLMs to integrate external knowledge flexibly through tool calling. In order to adapt the models to the new task, we construct a novel dataset TOOL-ED based on the EMPATHETICMPATHETIC DIALOGUE (ED) dataset. We validate EKTC on the ED dataset, and the experimental results demonstrate that our framework can enhance the ability of LLMs to generate empathetic responses effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01269v3">CPRM: A LLM-based Continual Pre-training Framework for Relevance Modeling in Commercial Search</a></div>
    <div class="paper-meta">
      📅 2024-12-08
    </div>
    <details class="paper-abstract">
      Relevance modeling between queries and items stands as a pivotal component in commercial search engines, directly affecting the user experience. Given the remarkable achievements of large language models (LLMs) in various natural language processing (NLP) tasks, LLM-based relevance modeling is gradually being adopted within industrial search systems. Nevertheless, foundational LLMs lack domain-specific knowledge and do not fully exploit the potential of in-context learning. Furthermore, structured item text remains underutilized, and there is a shortage in the supply of corresponding queries and background knowledge. We thereby propose CPRM (Continual Pre-training for Relevance Modeling), a framework designed for the continual pre-training of LLMs to address these issues. Our CPRM framework includes three modules: 1) employing both queries and multi-field item to jointly pre-train for enhancing domain knowledge, 2) applying in-context pre-training, a novel approach where LLMs are pre-trained on a sequence of related queries or items, and 3) conducting reading comprehension on items to produce associated domain knowledge and background information (e.g., generating summaries and corresponding queries) to further strengthen LLMs. Results on offline experiments and online A/B testing demonstrate that our model achieves convincing performance compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05896v1">XKV: Personalized KV Cache Memory Reduction for Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-12-08
    </div>
    <details class="paper-abstract">
      Recently the generative Large Language Model (LLM) has achieved remarkable success in numerous applications. Notably its inference generates output tokens one-by-one, leading to many redundant computations. The widely-used KV-Cache framework makes a compromise between time and space complexities. However, caching data generates the increasingly growing memory demand, that can quickly exhaust the limited memory capacity of the modern accelerator like GPUs, particularly in long-context inference tasks. Existing studies reduce memory consumption by evicting some of cached data that have less important impact on inference accuracy. But the benefit in practice is far from ideal due to the static cache allocation across different LLM network layers. This paper observes that the layer-specific cached data have very different impacts on accuracy. We quantify this difference, and give experimental and theoretical validation. We accordingly make a formal analysis and shows that customizing the cache size for each layer in a personalized manner can yield a significant memory reduction, while still providing comparable accuracy. We simulate the cache allocation as a combinatorial optimization problem and give a global optimal solution. In particular, we devise a mini- and sampling-based inference over a lightweight variant of the LLM model, so as to quickly capture the difference and then feed it into the personalized algorithms. Extensive experiments on real-world datasets demonstrate that our proposals can reduce KV cache memory consumption by 61.6% on average, improve computational efficiency by 2.1x and then increase the throughput by up to 5.5x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16698v4">Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-08
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      As AI systems pervade human life, ensuring that large language models (LLMs) make safe decisions remains a significant challenge. We introduce the Governance of the Commons Simulation (GovSim), a generative simulation platform designed to study strategic interactions and cooperative decision-making in LLMs. In GovSim, a society of AI agents must collectively balance exploiting a common resource with sustaining it for future use. This environment enables the study of how ethical considerations, strategic planning, and negotiation skills impact cooperative outcomes. We develop an LLM-based agent architecture and test it with the leading open and closed LLMs. We find that all but the most powerful LLM agents fail to achieve a sustainable equilibrium in GovSim, with the highest survival rate below 54%. Ablations reveal that successful multi-agent communication between agents is critical for achieving cooperation in these cases. Furthermore, our analyses show that the failure to achieve sustainable cooperation in most LLMs stems from their inability to formulate and analyze hypotheses about the long-term effects of their actions on the equilibrium of the group. Finally, we show that agents that leverage "Universalization"-based reasoning, a theory of moral thinking, are able to achieve significantly better sustainability. Taken together, GovSim enables us to study the mechanisms that underlie sustainable self-government with specificity and scale. We open source the full suite of our research results, including the simulation environment, agent prompts, and a comprehensive web interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10413v1">Evaluating Robustness of LLMs on Crisis-Related Microblogs across Events, Information Types, and Linguistic Features</a></div>
    <div class="paper-meta">
      📅 2024-12-08
      | 💬 12 pages, 10 figs, 5 tables
    </div>
    <details class="paper-abstract">
      The widespread use of microblogging platforms like X (formerly Twitter) during disasters provides real-time information to governments and response authorities. However, the data from these platforms is often noisy, requiring automated methods to filter relevant information. Traditionally, supervised machine learning models have been used, but they lack generalizability. In contrast, Large Language Models (LLMs) show better capabilities in understanding and processing natural language out of the box. This paper provides a detailed analysis of the performance of six well-known LLMs in processing disaster-related social media data from a large-set of real-world events. Our findings indicate that while LLMs, particularly GPT-4o and GPT-4, offer better generalizability across different disasters and information types, most LLMs face challenges in processing flood-related data, show minimal improvement despite the provision of examples (i.e., shots), and struggle to identify critical information categories like urgent requests and needs. Additionally, we examine how various linguistic features affect model performance and highlight LLMs' vulnerabilities against certain features like typos. Lastly, we provide benchmarking results for all events across both zero- and few-shot settings and observe that proprietary models outperform open-source ones in all tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05850v1">Cooperative SQL Generation for Segmented Databases By Using Multi-functional LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-08
    </div>
    <details class="paper-abstract">
      Text-to-SQL task aims to automatically yield SQL queries according to user text questions. To address this problem, we propose a Cooperative SQL Generation framework based on Multi-functional Agents (CSMA) through information interaction among large language model (LLM) based agents who own part of the database schema seperately. Inspired by the collaboration in human teamwork, CSMA consists of three stages: 1) Question-related schema collection, 2) Question-corresponding SQL query generation, and 3) SQL query correctness check. In the first stage, agents analyze their respective schema and communicate with each other to collect the schema information relevant to the question. In the second stage, agents try to generate the corresponding SQL query for the question using the collected information. In the third stage, agents check if the SQL query is created correctly according to their known information. This interaction-based method makes the question-relevant part of database schema from each agent to be used for SQL generation and check. Experiments on the Spider and Bird benckmark demonstrate that CSMA achieves a high performance level comparable to the state-of-the-arts, meanwhile holding the private data in these individual agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00800v2">A Comprehensive Guide to Explainable AI: From Classical Models to LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-08
    </div>
    <details class="paper-abstract">
      Explainable Artificial Intelligence (XAI) addresses the growing need for transparency and interpretability in AI systems, enabling trust and accountability in decision-making processes. This book offers a comprehensive guide to XAI, bridging foundational concepts with advanced methodologies. It explores interpretability in traditional models such as Decision Trees, Linear Regression, and Support Vector Machines, alongside the challenges of explaining deep learning architectures like CNNs, RNNs, and Large Language Models (LLMs), including BERT, GPT, and T5. The book presents practical techniques such as SHAP, LIME, Grad-CAM, counterfactual explanations, and causal inference, supported by Python code examples for real-world applications. Case studies illustrate XAI's role in healthcare, finance, and policymaking, demonstrating its impact on fairness and decision support. The book also covers evaluation metrics for explanation quality, an overview of cutting-edge XAI tools and frameworks, and emerging research directions, such as interpretability in federated learning and ethical AI considerations. Designed for a broad audience, this resource equips readers with the theoretical insights and practical skills needed to master XAI. Hands-on examples and additional resources are available at the companion GitHub repository: https://github.com/Echoslayer/XAI_From_Classical_Models_to_LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00774v5">Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-08
      | 💬 Project Page: https://freeze-omni.github.io/
    </div>
    <details class="paper-abstract">
      Rapidly developing large language models (LLMs) have brought tremendous intelligent applications. Especially, the GPT-4o's excellent duplex speech interaction ability has brought impressive experience to users. Researchers have recently proposed several multi-modal LLMs in this direction that can achieve user-agent speech-to-speech conversations. This paper proposes a novel speech-text multimodal LLM architecture called Freeze-Omni. Our main contribution is that the speech input and output modalities can be easily connected to a textual LLM while keeping the LLM's parameters frozen throughout the training process. We design a three-stage training strategy for modeling both the speech input and output, enabling Freeze-Omni to obtain speech-to-speech conversation ability using text-speech paired data (such as ASR and TTS data) and only 60,000 multi-round text Q&A data on 8 GPUs. Moreover, we can effectively ensure that the intelligence of the Freeze-Omni in the speech modality is at the same level compared with that in the text modality of its backbone LLM, while achieving low latency end-to-end spoken response. In addition, we also designed a method to achieve duplex dialogue ability through multi-task training, giving Freeze-Omni a more natural style of dialogue ability between users and agents. In summary, Freeze-Omni holds great potential to conduct speech-to-speech dialogue based on a multimodal LLM under the condition of a frozen LLM, avoiding the catastrophic forgetting problem caused by limited data and training resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11232v3">SLEGO: A Collaborative Data Analytics System with LLM Recommender for Diverse Users</a></div>
    <div class="paper-meta">
      📅 2024-12-08
      | 💬 14 pages, 9 figures
    </div>
    <details class="paper-abstract">
      This paper presents the SLEGO (Software-Lego) system, a collaborative analytics platform that bridges the gap between experienced developers and novice users using a cloud-based platform with modular, reusable microservices. These microservices enable developers to share their analytical tools and workflows, while a simple graphical user interface (GUI) allows novice users to build comprehensive analytics pipelines without programming skills. Supported by a knowledge base and a Large Language Model (LLM) powered recommendation system, SLEGO enhances the selection and integration of microservices, increasing the efficiency of analytics pipeline construction. Case studies in finance and machine learning illustrate how SLEGO promotes the sharing and assembly of modular microservices, significantly improving resource reusability and team collaboration. The results highlight SLEGO's role in democratizing data analytics by integrating modular design, knowledge bases, and recommendation systems, fostering a more inclusive and efficient analytical environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15296v2">MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-08
      | 💬 Produced by MME+MMBench+LLaVA Teams. Project Page: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Benchmarks
    </div>
    <details class="paper-abstract">
      As a prominent direction of Artificial General Intelligence (AGI), Multimodal Large Language Models (MLLMs) have garnered increased attention from both industry and academia. Building upon pre-trained LLMs, this family of models further develops multimodal perception and reasoning capabilities that are impressive, such as writing code given a flow chart or creating stories based on an image. In the development process, evaluation is critical since it provides intuitive feedback and guidance on improving models. Distinct from the traditional train-eval-test paradigm that only favors a single task like image classification, the versatility of MLLMs has spurred the rise of various new benchmarks and evaluation methods. In this paper, we aim to present a comprehensive survey of MLLM evaluation, discussing four key aspects: 1) the summarised benchmarks types divided by the evaluation capabilities, including foundation capabilities, model self-analysis, and extented applications; 2) the typical process of benchmark counstruction, consisting of data collection, annotation, and precautions; 3) the systematic evaluation manner composed of judge, metric, and toolkit; 4) the outlook for the next benchmark. This work aims to offer researchers an easy grasp of how to effectively evaluate MLLMs according to different needs and to inspire better evaluation methods, thereby driving the progress of MLLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06846v1">Classifier-free guidance in LLMs Safety</a></div>
    <div class="paper-meta">
      📅 2024-12-08
    </div>
    <details class="paper-abstract">
      The paper describes LLM unlearning without a retaining dataset, using the ORPO reinforcement learning method with inference enhanced by modified classifier-free guidance. Significant improvement in unlearning, without degradation of the model, is achieved through direct training on synthetic replacement data in CFG-aware training regime, with classifier-free guidance applied during the inference. This article is an extended version of the NeurIPS 2024 LLM-PC submission, which was awarded second prize.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05734v1">PrivAgent: Agentic-based Red-teaming for LLM Privacy Leakage</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      Recent studies have discovered that LLMs have serious privacy leakage concerns, where an LLM may be fooled into outputting private information under carefully crafted adversarial prompts. These risks include leaking system prompts, personally identifiable information, training data, and model parameters. Most existing red-teaming approaches for privacy leakage rely on humans to craft the adversarial prompts. A few automated methods are proposed for system prompt extraction, but they cannot be applied to more severe risks (e.g., training data extraction) and have limited effectiveness even for system prompt extraction. In this paper, we propose PrivAgent, a novel black-box red-teaming framework for LLM privacy leakage. We formulate different risks as a search problem with a unified attack goal. Our framework trains an open-source LLM through reinforcement learning as the attack agent to generate adversarial prompts for different target models under different risks. We propose a novel reward function to provide effective and fine-grained rewards for the attack agent. Finally, we introduce customizations to better fit our general framework to system prompt extraction and training data extraction. Through extensive evaluations, we first show that PrivAgent outperforms existing automated methods in system prompt leakage against six popular LLMs. Notably, our approach achieves a 100% success rate in extracting system prompts from real-world applications in OpenAI's GPT Store. We also show PrivAgent's effectiveness in extracting training data from an open-source LLM with a success rate of 5.9%. We further demonstrate PrivAgent's effectiveness in evading the existing guardrail defense and its helpfulness in enabling better safety alignment. Finally, we validate our customized designs through a detailed ablation study. We release our code here https://github.com/rucnyz/RedAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05631v1">CharacterBox: Evaluating the Role-Playing Capabilities of LLMs in Text-Based Virtual Worlds</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      Role-playing is a crucial capability of Large Language Models (LLMs), enabling a wide range of practical applications, including intelligent non-player characters, digital twins, and emotional companions. Evaluating this capability in LLMs is challenging due to the complex dynamics involved in role-playing, such as maintaining character fidelity throughout a storyline and navigating open-ended narratives without a definitive ground truth. Current evaluation methods, which primarily focus on question-answering or conversational snapshots, fall short of adequately capturing the nuanced character traits and behaviors essential for authentic role-playing. In this paper, we propose CharacterBox, which is a simulation sandbox designed to generate situational fine-grained character behavior trajectories. These behavior trajectories enable a more comprehensive and in-depth evaluation of role-playing capabilities. CharacterBox consists of two main components: the character agent and the narrator agent. The character agent, grounded in psychological and behavioral science, exhibits human-like behaviors, while the narrator agent coordinates interactions between character agents and environmental changes. Additionally, we introduce two trajectory-based methods that leverage CharacterBox to enhance LLM performance. To reduce costs and facilitate the adoption of CharacterBox by public communities, we fine-tune two smaller models, CharacterNR and CharacterRM, as substitutes for GPT API calls, and demonstrate their competitive performance compared to advanced GPT APIs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13889v2">LLM with Relation Classifier for Document-Level Relation Extraction</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have created a new paradigm for natural language processing. Despite their advancement, LLM-based methods still lag behind traditional approaches in document-level relation extraction (DocRE), a critical task for understanding complex entity relations within long context. This paper investigates the causes of this performance gap, identifying the dispersion of attention by LLMs due to entity pairs without relations as a key factor. We then introduce a novel classifier-LLM approach to DocRE. Particularly, the proposed approach begins with a classifier designed to select entity pair candidates that exhibit potential relations and then feed them to LLM for final relation classification. This method ensures that the LLM's attention is directed at relation-expressing entity pairs instead of those without relations during inference. Experiments on DocRE benchmarks reveal that our method significantly outperforms recent LLM-based DocRE models and narrows the performance gap with state-of-the-art BERT-based models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02391v2">Scaling Laws for Economic Productivity: Experimental Evidence in LLM-Assisted Translation</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      This paper derives "scaling laws"--empirical relationships between the training compute of Large Language Models (LLMs) and their performance--for economic outcomes. In a preregistered online experiment, 300 professional translators completed 1,800 tasks using one of 13 LLMs (or a control). A tenfold increase in model compute improved task completion speed by 12.3%, grades by 0.18 standard deviations, and earnings per minute by 16.1%. Gains were four times larger for lower-skilled workers. These findings suggest continued model scaling could boost U.S. productivity by at least 6.9% over the next decade.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05586v1">Towards Learning to Reason: Comparing LLMs with Neuro-Symbolic on Arithmetic Relations in Abstract Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      This work compares large language models (LLMs) and neuro-symbolic approaches in solving Raven's progressive matrices (RPM), a visual abstract reasoning test that involves the understanding of mathematical rules such as progression or arithmetic addition. Providing the visual attributes directly as textual prompts, which assumes an oracle visual perception module, allows us to measure the model's abstract reasoning capability in isolation. Despite providing such compositionally structured representations from the oracle visual perception and advanced prompting techniques, both GPT-4 and Llama-3 70B cannot achieve perfect accuracy on the center constellation of the I-RAVEN dataset. Our analysis reveals that the root cause lies in the LLM's weakness in understanding and executing arithmetic rules. As a potential remedy, we analyze the Abductive Rule Learner with Context-awareness (ARLC), a neuro-symbolic approach that learns to reason with vector-symbolic architectures (VSAs). Here, concepts are represented with distributed vectors s.t. dot products between encoded vectors define a similarity kernel, and simple element-wise operations on the vectors perform addition/subtraction on the encoded values. We find that ARLC achieves almost perfect accuracy on the center constellation of I-RAVEN, demonstrating a high fidelity in arithmetic rules. To stress the length generalization capabilities of the models, we extend the RPM tests to larger matrices (3x10 instead of typical 3x3) and larger dynamic ranges of the attribute values (from 10 up to 1000). We find that the LLM's accuracy of solving arithmetic rules drops to sub-10%, especially as the dynamic range expands, while ARLC can maintain a high accuracy due to emulating symbolic computations on top of properly distributed representations. Our code is available at https://github.com/IBM/raven-large-language-models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05561v1">Exploring the Use of LLMs for SQL Equivalence Checking</a></div>
    <div class="paper-meta">
      📅 2024-12-07
    </div>
    <details class="paper-abstract">
      Equivalence checking of two SQL queries is an intractable problem encountered in diverse contexts ranging from grading student submissions in a DBMS course to debugging query rewriting rules in an optimizer, and many more. While a lot of progress has been made in recent years in developing practical solutions for this problem, the existing methods can handle only a small subset of SQL, even for bounded equivalence checking. They cannot support sophisticated SQL expressions one encounters in practice. At the same time, large language models (LLMs) -- such as GPT-4 -- have emerged as power generators of SQL from natural language specifications. This paper explores whether LLMs can also demonstrate the ability to reason with SQL queries and help advance SQL equivalence checking. Towards this, we conducted a detailed evaluation of several LLMs over collections with SQL pairs of varying levels of complexity. We explored the efficacy of different prompting techniques, the utility of synthetic examples & explanations, as well as logical plans generated by query parsers. Our main finding is that with well-designed prompting using an unoptimized SQL Logical Plan, LLMs can perform equivalence checking beyond the capabilities of current techniques, achieving nearly 100% accuracy for equivalent pairs and up to 70% for non-equivalent pairs of SQL queries. While LLMs lack the ability to generate formal proofs, their synthetic examples and human-readable explanations offer valuable insights to students (& instructors) in a classroom setting and to database administrators (DBAs) managing large database installations. Additionally, we also show that with careful fine-tuning, we can close the performance gap between smaller (and efficient) models and larger models such as GPT, thus paving the way for potential LLM-integration in standalone data processing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00539v2">TextClass Benchmark: A Continuous Elo Rating of LLMs in Social Sciences</a></div>
    <div class="paper-meta">
      📅 2024-12-07
      | 💬 Working paper: 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The TextClass Benchmark project is an ongoing, continuous benchmarking process that aims to provide a comprehensive, fair, and dynamic evaluation of LLMs and transformers for text classification tasks. This evaluation spans various domains and languages in social sciences disciplines engaged in NLP and text-as-data approach. The leaderboards present performance metrics and relative ranking using a tailored Elo rating system. With each leaderboard cycle, novel models are added, fixed test sets can be replaced for unseen, equivalent data to test generalisation power, ratings are updated, and a Meta-Elo leaderboard combines and weights domain-specific leaderboards. This article presents the rationale and motivation behind the project, explains the Elo rating system in detail, and estimates Meta-Elo across different classification tasks in social science disciplines. We also present a snapshot of the first cycle of classification tasks on incivility data in Chinese, English, German and Russian. This ongoing benchmarking process includes not only additional languages such as Arabic, Hindi, and Spanish but also a classification of policy agenda topics, misinformation, among others.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06828v1">Enhancing LLMs for Impression Generation in Radiology Reports through a Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      This study introduces "RadCouncil," a multi-agent Large Language Model (LLM) framework designed to enhance the generation of impressions in radiology reports from the finding section. RadCouncil comprises three specialized agents: 1) a "Retrieval" Agent that identifies and retrieves similar reports from a vector database, 2) a "Radiologist" Agent that generates impressions based on the finding section of the given report plus the exemplar reports retrieved by the Retrieval Agent, and 3) a "Reviewer" Agent that evaluates the generated impressions and provides feedback. The performance of RadCouncil was evaluated using both quantitative metrics (BLEU, ROUGE, BERTScore) and qualitative criteria assessed by GPT-4, using chest X-ray as a case study. Experiment results show improvements in RadCouncil over the single-agent approach across multiple dimensions, including diagnostic accuracy, stylistic concordance, and clarity. This study highlights the potential of utilizing multiple interacting LLM agents, each with a dedicated task, to enhance performance in specialized medical tasks and the development of more robust and adaptable healthcare AI solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13713v2">Comparison of Open-Source and Proprietary LLMs for Machine Reading Comprehension: A Practical Analysis for Industrial Applications</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 Preprint submitted to Natural Language Processing Journal
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated remarkable performance in various Natural Language Processing (NLP) applications, such as sentiment analysis, content generation, and personalized recommendations. Despite their impressive capabilities, there remains a significant need for systematic studies concerning the practical application of LLMs in industrial settings, as well as the specific requirements and challenges related to their deployment in these contexts. This need is particularly critical for Machine Reading Comprehension (MCR), where factual, concise, and accurate responses are required. To date, most MCR rely on Small Language Models (SLMs) or Recurrent Neural Networks (RNNs) such as Long Short-Term Memory (LSTM). This trend is evident in the SQuAD2.0 rankings on the Papers with Code table. This article presents a comparative analysis between open-source LLMs and proprietary models on this task, aiming to identify light and open-source alternatives that offer comparable performance to proprietary models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06827v1">Enhancing LLMs for Physics Problem-Solving using Reinforcement Learning with Human-AI Feedback</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in text-based tasks but struggle with the complex reasoning required for physics problems, particularly in advanced arithmetic and conceptual understanding. While some research has explored ways to enhance LLMs in physics education using techniques such as prompt engineering and Retrieval Augmentation Generation (RAG), not enough effort has been made in addressing their limitations in physics reasoning. This paper presents a novel approach to improving LLM performance on physics questions using Reinforcement Learning with Human and Artificial Intelligence Feedback (RLHAIF). We evaluate several reinforcement learning methods, including Proximal Policy Optimization (PPO), Direct Preference Optimization (DPO), and Remax optimization. These methods are chosen to investigate RL policy performance with different settings on the PhyQA dataset, which includes challenging physics problems from high school textbooks. Our RLHAIF model, tested on leading LLMs like LLaMA2 and Mistral, achieved superior results, notably with the MISTRAL-PPO model, demonstrating marked improvements in reasoning and accuracy. It achieved high scores, with a 58.67 METEOR score and a 0.74 Reasoning score, making it a strong example for future physics reasoning research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05393v1">HiVeGen -- Hierarchical LLM-based Verilog Generation for Scalable Chip Design</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      With Large Language Models (LLMs) recently demonstrating impressive proficiency in code generation, it is promising to extend their abilities to Hardware Description Language (HDL). However, LLMs tend to generate single HDL code blocks rather than hierarchical structures for hardware designs, leading to hallucinations, particularly in complex designs like Domain-Specific Accelerators (DSAs). To address this, we propose HiVeGen, a hierarchical LLM-based Verilog generation framework that decomposes generation tasks into LLM-manageable hierarchical submodules. HiVeGen further harnesses the advantages of such hierarchical structures by integrating automatic Design Space Exploration (DSE) into hierarchy-aware prompt generation, introducing weight-based retrieval to enhance code reuse, and enabling real-time human-computer interaction to lower error-correction cost, significantly improving the quality of generated designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05232v1">LIAR: Leveraging Alignment (Best-of-N) to Jailbreak LLMs in Seconds</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Many existing jailbreak techniques rely on solving discrete combinatorial optimization, while more recent approaches involve training LLMs to generate multiple adversarial prompts. However, both approaches require significant computational resources to produce even a single adversarial prompt. We hypothesize that the inefficiency of current approaches stems from an inadequate characterization of the jailbreak problem. To address this gap, we formulate the jailbreak problem in terms of alignment. By starting from an available safety-aligned model, we leverage an unsafe reward to guide the safe model towards generating unsafe outputs using alignment techniques (e.g., reinforcement learning from human feedback), effectively performing jailbreaking via alignment. We propose a novel jailbreak method called LIAR (LeveragIng Alignment to jailbReak). To demonstrate the simplicity and effectiveness of our approach, we employ a best-of-N method to solve the alignment problem. LIAR offers significant advantages: lower computational requirements without additional training, fully black-box operation, competitive attack success rates, and more human-readable prompts. We provide theoretical insights into the possibility of jailbreaking a safety-aligned model, revealing inherent vulnerabilities in current alignment strategies for LLMs. We also provide sub-optimality guarantees for the proposed \algo. Experimentally, we achieve ASR comparable to the SoTA with a 10x improvement to perplexity and a Time-to-Attack measured in seconds rather than tens of hours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05206v1">ConQRet: Benchmarking Fine-Grained Evaluation of Retrieval Augmented Argumentation with LLM Judges</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Computational argumentation, which involves generating answers or summaries for controversial topics like abortion bans and vaccination, has become increasingly important in today's polarized environment. Sophisticated LLM capabilities offer the potential to provide nuanced, evidence-based answers to such questions through Retrieval-Augmented Argumentation (RAArg), leveraging real-world evidence for high-quality, grounded arguments. However, evaluating RAArg remains challenging, as human evaluation is costly and difficult for complex, lengthy answers on complicated topics. At the same time, re-using existing argumentation datasets is no longer sufficient, as they lack long, complex arguments and realistic evidence from potentially misleading sources, limiting holistic evaluation of retrieval effectiveness and argument quality. To address these gaps, we investigate automated evaluation methods using multiple fine-grained LLM judges, providing better and more interpretable assessments than traditional single-score metrics and even previously reported human crowdsourcing. To validate the proposed techniques, we introduce ConQRet, a new benchmark featuring long and complex human-authored arguments on debated topics, grounded in real-world websites, allowing an exhaustive evaluation across retrieval effectiveness, argument quality, and groundedness. We validate our LLM Judges on a prior dataset and the new ConQRet benchmark. Our proposed LLM Judges and the ConQRet benchmark can enable rapid progress in computational argumentation and can be naturally extended to other complex retrieval-augmented generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03019v2">Is Your Paper Being Reviewed by an LLM? Investigating AI Text Detectability in Peer Review</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Peer review is a critical process for ensuring the integrity of published scientific research. Confidence in this process is predicated on the assumption that experts in the relevant domain give careful consideration to the merits of manuscripts which are submitted for publication. With the recent rapid advancements in the linguistic capabilities of large language models (LLMs), a new potential risk to the peer review process is that negligent reviewers will rely on LLMs to perform the often time consuming process of reviewing a paper. In this study, we investigate the ability of existing AI text detection algorithms to distinguish between peer reviews written by humans and different state-of-the-art LLMs. Our analysis shows that existing approaches fail to identify many GPT-4o written reviews without also producing a high number of false positive classifications. To address this deficiency, we propose a new detection approach which surpasses existing methods in the identification of GPT-4o written peer reviews at low levels of false positive classifications. Our work reveals the difficulty of accurately identifying AI-generated text at the individual review level, highlighting the urgent need for new tools and methods to detect this type of unethical application of generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06467v2">WAPITI: A Watermark for Finetuned Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Watermarking of large language models (LLMs) generation embeds an imperceptible statistical pattern within texts, making it algorithmically detectable. Watermarking is a promising method for addressing potential harm and biases from LLMs, as it enables traceability, accountability, and detection of manipulated content, helping to mitigate unintended consequences. However, for open-source models, watermarking faces two major challenges: (i) incompatibility with fine-tuned models, and (ii) vulnerability to fine-tuning attacks. In this work, we propose WAPITI, a new method that transfers watermarking from base models to fine-tuned models through parameter integration. To the best of our knowledge, we propose the first watermark for fine-tuned open-source LLMs that preserves their fine-tuned capabilities. Furthermore, our approach offers an effective defense against fine-tuning attacks. We test our method on various model architectures and watermarking strategies. Results demonstrate that our method can successfully inject watermarks and is highly compatible with fine-tuned models. Additionally, we offer an in-depth analysis of how parameter editing influences the watermark strength and overall capabilities of the resulting models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05153v1">A text-to-tabular approach to generate synthetic patient data using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 12 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Access to large-scale high-quality healthcare databases is key to accelerate medical research and make insightful discoveries about diseases. However, access to such data is often limited by patient privacy concerns, data sharing restrictions and high costs. To overcome these limitations, synthetic patient data has emerged as an alternative. However, synthetic data generation (SDG) methods typically rely on machine learning (ML) models trained on original data, leading back to the data scarcity problem. We propose an approach to generate synthetic tabular patient data that does not require access to the original data, but only a description of the desired database. We leverage prior medical knowledge and in-context learning capabilities of large language models (LLMs) to generate realistic patient data, even in a low-resource setting. We quantitatively evaluate our approach against state-of-the-art SDG models, using fidelity, privacy, and utility metrics. Our results show that while LLMs may not match the performance of state-of-the-art models trained on the original data, they effectively generate realistic patient data with well-preserved clinical correlations. An ablation study highlights key elements of our prompt contributing to high-quality synthetic patient data generation. This approach, which is easy to use and does not require original data or advanced ML skills, is particularly valuable for quickly generating custom-designed patient data, supporting project implementation and providing educational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05250v3">LLM-Enhanced Bayesian Optimization for Efficient Analog Layout Constraint Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Analog layout synthesis faces significant challenges due to its dependence on manual processes, considerable time requirements, and performance instability. Current Bayesian Optimization (BO)-based techniques for analog layout synthesis, despite their potential for automation, suffer from slow convergence and extensive data needs, limiting their practical application. This paper presents the \texttt{LLANA} framework, a novel approach that leverages Large Language Models (LLMs) to enhance BO by exploiting the few-shot learning abilities of LLMs for more efficient generation of analog design-dependent parameter constraints. Experimental results demonstrate that \texttt{LLANA} not only achieves performance comparable to state-of-the-art (SOTA) BO methods but also enables a more effective exploration of the analog circuit design space, thanks to LLM's superior contextual understanding and learning efficiency. The code is available at https://github.com/dekura/LLANA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02976v2">Hallucination Detection in LLMs: Fast and Memory-Efficient Fine-Tuned Models</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Uncertainty estimation is a necessary component when implementing AI in high-risk settings, such as autonomous cars, medicine, or insurances. Large Language Models (LLMs) have seen a surge in popularity in recent years, but they are subject to hallucinations, which may cause serious harm in high-risk settings. Despite their success, LLMs are expensive to train and run: they need a large amount of computations and memory, preventing the use of ensembling methods in practice. In this work, we present a novel method that allows for fast and memory-friendly training of LLM ensembles. We show that the resulting ensembles can detect hallucinations and are a viable approach in practice as only one GPU is needed for training and inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09439v2">Towards Boosting LLMs-driven Relevance Modeling with Progressive Retrieved Behavior-augmented Prompting</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 Accepted By COLING 2025
    </div>
    <details class="paper-abstract">
      Relevance modeling is a critical component for enhancing user experience in search engines, with the primary objective of identifying items that align with users' queries. Traditional models only rely on the semantic congruence between queries and items to ascertain relevance. However, this approach represents merely one aspect of the relevance judgement, and is insufficient in isolation. Even powerful Large Language Models (LLMs) still cannot accurately judge the relevance of a query and an item from a semantic perspective. To augment LLMs-driven relevance modeling, this study proposes leveraging user interactions recorded in search logs to yield insights into users' implicit search intentions. The challenge lies in the effective prompting of LLMs to capture dynamic search intentions, which poses several obstacles in real-world relevance scenarios, i.e., the absence of domain-specific knowledge, the inadequacy of an isolated prompt, and the prohibitive costs associated with deploying LLMs. In response, we propose ProRBP, a novel Progressive Retrieved Behavior-augmented Prompting framework for integrating search scenario-oriented knowledge with LLMs effectively. Specifically, we perform the user-driven behavior neighbors retrieval from the daily search logs to obtain domain-specific knowledge in time, retrieving candidates that users consider to meet their expectations. Then, we guide LLMs for relevance modeling by employing advanced prompting techniques that progressively improve the outputs of the LLMs, followed by a progressive aggregation with comprehensive consideration of diverse aspects. For online serving, we have developed an industrial application framework tailored for the deployment of LLMs in relevance modeling. Experiments on real-world industry data and online A/B testing demonstrate our proposal achieves promising performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01349v3">LLMs May Perform MCQA by Selecting the Least Incorrect Option</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 COLING 2025
    </div>
    <details class="paper-abstract">
      In the field of NLP, Large Language Models (LLMs) have markedly enhanced performance across a variety of tasks. However, the comprehensive evaluation of LLMs remains an inevitable challenge for the community. Recently, the adoption of Multiple Choice Question Answering (MCQA) as a benchmark for assessing LLMs has gained considerable traction. However, concerns regarding the robustness of this evaluative method persist. Building upon previous discussions on the issue of \textit{variability}, we reveal an additional dimension of concern: LLMs may perform MCQA by selecting the least incorrect option rather than distinctly correct. This observation suggests that LLMs might regard multiple options as correct, which could undermine the reliability of MCQA as a metric for evaluating LLMs. To address this challenge, we introduce an enhanced dataset augmentation method for MCQA, termed MCQA+, to provide a more accurate reflection of the model performance, thereby highlighting the necessity for more sophisticated evaluation mechanisms in the assessment of LLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04315v2">Densing Law of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a milestone in artificial intelligence, and their performance can improve as the model size increases. However, this scaling brings great challenges to training and inference efficiency, particularly for deploying LLMs in resource-constrained environments, and the scaling trend is becoming increasingly unsustainable. This paper introduces the concept of ``\textit{capacity density}'' as a new metric to evaluate the quality of the LLMs across different scales and describes the trend of LLMs in terms of both effectiveness and efficiency. To calculate the capacity density of a given target LLM, we first introduce a set of reference models and develop a scaling law to predict the downstream performance of these reference models based on their parameter sizes. We then define the \textit{effective parameter size} of the target LLM as the parameter size required by a reference model to achieve equivalent performance, and formalize the capacity density as the ratio of the effective parameter size to the actual parameter size of the target LLM. Capacity density provides a unified framework for assessing both model effectiveness and efficiency. Our further analysis of recent open-source base LLMs reveals an empirical law (the densing law)that the capacity density of LLMs grows exponentially over time. More specifically, using some widely used benchmarks for evaluation, the capacity density of LLMs doubles approximately every three months. The law provides new perspectives to guide future LLM development, emphasizing the importance of improving capacity density to achieve optimal results with minimal computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04917v1">Continuous Speech Tokens Makes LLMs Robust Multi-Modality Learners</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Recent advances in GPT-4o like multi-modality models have demonstrated remarkable progress for direct speech-to-speech conversation, with real-time speech interaction experience and strong speech understanding ability. However, current research focuses on discrete speech tokens to align with discrete text tokens for language modelling, which depends on an audio codec with residual connections or independent group tokens, such a codec usually leverages large scale and diverse datasets training to ensure that the discrete speech codes have good representation for varied domain, noise, style data reconstruction as well as a well-designed codec quantizer and encoder-decoder architecture for discrete token language modelling. This paper introduces Flow-Omni, a continuous speech token based GPT-4o like model, capable of real-time speech interaction and low streaming latency. Specifically, first, instead of cross-entropy loss only, we combine flow matching loss with a pretrained autoregressive LLM and a small MLP network to predict the probability distribution of the continuous-valued speech tokens from speech prompt. second, we incorporated the continuous speech tokens to Flow-Omni multi-modality training, thereby achieving robust speech-to-speech performance with discrete text tokens and continuous speech tokens together. Experiments demonstrate that, compared to discrete text and speech multi-modality training and its variants, the continuous speech tokens mitigate robustness issues by avoiding the inherent flaws of discrete speech code's representation loss for LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04871v1">Building a Family of Data Augmentation Models for Low-cost LLM Fine-tuning on the Cloud</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 coling 2025 industry track
    </div>
    <details class="paper-abstract">
      Specializing LLMs in various domain-specific tasks has emerged as a critical step towards achieving high performance. However, the construction and annotation of datasets in specific domains are always very costly. Apart from using superior and expensive closed-source LLM APIs to construct datasets, some open-source models have become strong enough to handle dataset construction in many scenarios. Thus, we present a family of data augmentation models designed to significantly improve the efficiency for model fine-tuning. These models, trained based on sufficiently small LLMs, support key functionalities with low inference costs: instruction expansion, instruction refinement, and instruction-response pair expansion. To fulfill this goal, we first construct an automatic data collection system with seed datasets generated from both public repositories and our in-house datasets. This system leverages powerful LLMs to expand, refine and re-write the instructions and responses, incorporating quality assessment techniques. Following this, we introduce the training process of our models, which effectively distills task-solving and text synthesis abilities from teacher LLMs. Finally, we demonstrate how we integrate these functionalities into a machine learning platform to support low-cost LLM fine-tuning from both dataset preparation and training perspectives for users. Experiments and an application study prove the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02580v2">PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 Accepted by NDSS Symposium 2025. Please cite the conference version of this paper, e.g., "Ye Liu, Yue Xue, Daoyuan Wu, Yuqiang Sun, Yi Li, Miaolei Shi, Yang Liu. PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation. In 32nd Annual Network and Distributed System Security Symposium (NDSS 2025)."
    </div>
    <details class="paper-abstract">
      With recent advances in large language models (LLMs), this paper explores the potential of leveraging state-of-the-art LLMs,such as GPT-4, to transfer existing human-written properties (e.g.,those from Certora auditing reports) and automatically generate customized properties for unknown code. To this end, we embed existing properties into a vector database and retrieve a reference property for LLM-based in-context learning to generate a new property for a given code. While this basic process is relatively straightforward, ensuring that the generated properties are (i) compilable, (ii) appropriate, and (iii) verifiable presents challenges. To address (i), we use the compilation and static analysis feedback as an external oracle to guide LLMs in iteratively revising the generated properties. For (ii), we consider multiple dimensions of similarity to rank the properties and employ a weighted algorithm to identify the top-K properties as the final result. For (iii), we design a dedicated prover to formally verify the correctness of the generated properties. We have implemented these strategies into a novel LLM-based property generation tool called PropertyGPT. Our experiments show that PropertyGPT can generate comprehensive and high-quality properties, achieving an 80% recall compared to the ground truth. It successfully detected 26 CVEs/attack incidents out of 37 tested and also uncovered 12 zero-day vulnerabilities, leading to $8,256 in bug bounty rewards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05721v3">PsycoLLM: Enhancing LLM for Psychological Understanding and Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-12-06
      | 💬 Accepted by IEEE Transactions on Computational Social Systems. https://github.com/MACLAB-HFUT/PsycoLLM
    </div>
    <details class="paper-abstract">
      Mental health has attracted substantial attention in recent years and LLM can be an effective technology for alleviating this problem owing to its capability in text understanding and dialogue. However, existing research in this domain often suffers from limitations, such as training on datasets lacking crucial prior knowledge and evidence, and the absence of comprehensive evaluation methods. In this paper, we propose a specialized psychological large language model (LLM), named PsycoLLM, trained on a proposed high-quality psychological dataset, including single-turn QA, multi-turn dialogues and knowledge-based QA. Specifically, we construct multi-turn dialogues through a three-step pipeline comprising multi-turn QA generation, evidence judgment, and dialogue refinement. We augment this process with real-world psychological case backgrounds extracted from online platforms, enhancing the relevance and applicability of the generated data. Additionally, to compare the performance of PsycoLLM with other LLMs, we develop a comprehensive psychological benchmark based on authoritative psychological counseling examinations in China, which includes assessments of professional ethics, theoretical proficiency, and case analysis. The experimental results on the benchmark illustrate the effectiveness of PsycoLLM, which demonstrates superior performance compared to other LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04806v1">Rethinking Time Series Forecasting with LLMs via Nearest Neighbor Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Adapting Large Language Models (LLMs) that are extensively trained on abundant text data, and customizing the input prompt to enable time series forecasting has received considerable attention. While recent work has shown great potential for adapting the learned prior of LLMs, the formulation of the prompt to finetune LLMs remains challenging as prompt should be aligned with time series data. Additionally, current approaches do not effectively leverage word token embeddings which embody the rich representation space learned by LLMs. This emphasizes the need for a robust approach to formulate the prompt which utilizes the word token embeddings while effectively representing the characteristics of the time series. To address these challenges, we propose NNCL-TLLM: Nearest Neighbor Contrastive Learning for Time series forecasting via LLMs. First, we generate time series compatible text prototypes such that each text prototype represents both word token embeddings in its neighborhood and time series characteristics via end-to-end finetuning. Next, we draw inspiration from Nearest Neighbor Contrastive Learning to formulate the prompt while obtaining the top-$k$ nearest neighbor time series compatible text prototypes. We then fine-tune the layer normalization and positional embeddings of the LLM, keeping the other layers intact, reducing the trainable parameters and decreasing the computational cost. Our comprehensive experiments demonstrate that NNCL-TLLM outperforms in few-shot forecasting while achieving competitive or superior performance over the state-of-the-art methods in long-term and short-term forecasting tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01672v3">Practicing Stress Relief for the Everyday: Designing Social Simulation Using VR, AR, and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      Stress is an inevitable part of day-to-day life yet many find themselves unable to manage it themselves, particularly when professional or peer support are not always readily available. As self-care becomes increasingly vital for mental well-being, this paper explores the potential of social simulation as a safe, virtual environment for practicing stress relief for everyday situations. Leveraging the immersive capabilities of VR, AR, and LLMs, we developed eight interactive prototypes for various everyday stressful scenarios (e.g. public speaking) then conducted prototype-driven semi-structured interviews with 19 participants. We reveal that people currently lack effective means to support themselves through everyday stress and found that social simulation fills a gap for simulating real environments for training mental health practices. We outline key considerations for future development of simulation for self-care, including risks of trauma from hyper-realism, distrust of LLM-recommended timing for mental health recommendations, and the value of accessibility for self-care interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04757v1">Ltri-LLM: Streaming Long Context Inference for LLMs with Training-Free Dynamic Triangular Attention Pattern</a></div>
    <div class="paper-meta">
      📅 2024-12-06
    </div>
    <details class="paper-abstract">
      The quadratic computational complexity of the attention mechanism in current Large Language Models (LLMs) renders inference with long contexts prohibitively expensive. To address this challenge, various approaches aim to retain critical portions of the context to optimally approximate Full Attention (FA) through Key-Value (KV) compression or Sparse Attention (SA), enabling the processing of virtually unlimited text lengths in a streaming manner. However, these methods struggle to achieve performance levels comparable to FA, particularly in retrieval tasks. In this paper, our analysis of attention head patterns reveals that LLMs' attention distributions show strong local correlations, naturally reflecting a chunking mechanism for input context. We propose Ltri-LLM framework, which divides KVs into spans, stores them in an offline index, and retrieves the relevant KVs into memory for various queries. Experimental results on popular long text benchmarks show that Ltri-LLM can achieve performance close to FA while maintaining efficient, streaming-based inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04642v1">Improving LLM Group Fairness on Tabular Data via In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to be effective on tabular prediction tasks in the low-data regime, leveraging their internal knowledge and ability to learn from instructions and examples. However, LLMs can fail to generate predictions that satisfy group fairness, that is, produce equitable outcomes across groups. Critically, conventional debiasing approaches for natural language tasks do not directly translate to mitigating group unfairness in tabular settings. In this work, we systematically investigate four empirical approaches to improve group fairness of LLM predictions on tabular datasets, including fair prompt optimization, soft prompt tuning, strategic selection of few-shot examples, and self-refining predictions via chain-of-thought reasoning. Through experiments on four tabular datasets using both open-source and proprietary LLMs, we show the effectiveness of these methods in enhancing demographic parity while maintaining high overall performance. Our analysis provides actionable insights for practitioners in selecting the most suitable approach based on their specific requirements and constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10361v1">How Large Language Models (LLMs) Extrapolate: From Guided Missiles to Guided Prompts</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      This paper argues that we should perceive LLMs as machines of extrapolation. Extrapolation is a statistical function for predicting the next value in a series. Extrapolation contributes to both GPT successes and controversies surrounding its hallucination. The term hallucination implies a malfunction, yet this paper contends that it in fact indicates the chatbot efficiency in extrapolation, albeit an excess of it. This article bears a historical dimension: it traces extrapolation to the nascent years of cybernetics. In 1941, when Norbert Wiener transitioned from missile science to communication engineering, the pivotal concept he adopted was none other than extrapolation. Soviet mathematician Andrey Kolmogorov, renowned for his compression logic that inspired OpenAI, had developed in 1939 another extrapolation project that Wiener later found rather like his own. This paper uncovers the connections between hot war science, Cold War cybernetics, and the contemporary debates on LLM performances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04576v1">Show, Don't Tell: Uncovering Implicit Character Portrayal using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      Tools for analyzing character portrayal in fiction are valuable for writers and literary scholars in developing and interpreting compelling stories. Existing tools, such as visualization tools for analyzing fictional characters, primarily rely on explicit textual indicators of character attributes. However, portrayal is often implicit, revealed through actions and behaviors rather than explicit statements. We address this gap by leveraging large language models (LLMs) to uncover implicit character portrayals. We start by generating a dataset for this task with greater cross-topic similarity, lexical diversity, and narrative lengths than existing narrative text corpora such as TinyStories and WritingPrompts. We then introduce LIIPA (LLMs for Inferring Implicit Portrayal for Character Analysis), a framework for prompting LLMs to uncover character portrayals. LIIPA can be configured to use various types of intermediate computation (character attribute word lists, chain-of-thought) to infer how fictional characters are portrayed in the source text. We find that LIIPA outperforms existing approaches, and is more robust to increasing character counts (number of unique persons depicted) due to its ability to utilize full narrative context. Lastly, we investigate the sensitivity of portrayal estimates to character demographics, identifying a fairness-accuracy tradeoff among methods in our LIIPA framework -- a phenomenon familiar within the algorithmic fairness literature. Despite this tradeoff, all LIIPA variants consistently outperform non-LLM baselines in both fairness and accuracy. Our work demonstrates the potential benefits of using LLMs to analyze complex characters and to better understand how implicit portrayal biases may manifest in narrative texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04415v1">Targeting the Core: A Simple and Effective Method to Attack RAG-based Agents via Direct LLM Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      AI agents, powered by large language models (LLMs), have transformed human-computer interactions by enabling seamless, natural, and context-aware communication. While these advancements offer immense utility, they also inherit and amplify inherent safety risks such as bias, fairness, hallucinations, privacy breaches, and a lack of transparency. This paper investigates a critical vulnerability: adversarial attacks targeting the LLM core within AI agents. Specifically, we test the hypothesis that a deceptively simple adversarial prefix, such as \textit{Ignore the document}, can compel LLMs to produce dangerous or unintended outputs by bypassing their contextual safeguards. Through experimentation, we demonstrate a high attack success rate (ASR), revealing the fragility of existing LLM defenses. These findings emphasize the urgent need for robust, multi-layered security measures tailored to mitigate vulnerabilities at the LLM level and within broader agent-based architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04318v1">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 Under review at ICLR
    </div>
    <details class="paper-abstract">
      This paper introduces the counter-intuitive generalization results of overfitting pre-trained large language models (LLMs) on very small datasets. In the setting of open-ended text generation, it is well-documented that LLMs tend to generate repetitive and dull sequences, a phenomenon that is especially apparent when generating using greedy decoding. This issue persists even with state-of-the-art LLMs containing billions of parameters, trained via next-token prediction on large datasets. We find that by further fine-tuning these models to achieve a near-zero training loss on a small set of samples -- a process we refer to as hyperfitting -- the long-sequence generative capabilities are greatly enhanced. Greedy decoding with these Hyperfitted models even outperform Top-P sampling over long-sequences, both in terms of diversity and human preferences. This phenomenon extends to LLMs of various sizes, different domains, and even autoregressive image generation. We further find this phenomena to be distinctly different from that of Grokking and double descent. Surprisingly, our experiments indicate that hyperfitted models rarely fall into repeating sequences they were trained on, and even explicitly blocking these sequences results in high-quality output. All hyperfitted models produce extremely low-entropy predictions, often allocating nearly all probability to a single token.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04235v1">Addressing Hallucinations with RAG and NMISS in Italian Healthcare LLM Chatbots</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      I combine detection and mitigation techniques to addresses hallucinations in Large Language Models (LLMs). Mitigation is achieved in a question-answering Retrieval-Augmented Generation (RAG) framework while detection is obtained by introducing the Negative Missing Information Scoring System (NMISS), which accounts for contextual relevance in responses. While RAG mitigates hallucinations by grounding answers in external data, NMISS refines the evaluation by identifying cases where traditional metrics incorrectly flag contextually accurate responses as hallucinations. I use Italian health news articles as context to evaluate LLM performance. Results show that Gemma2 and GPT-4 outperform the other models, with GPT-4 producing answers closely aligned with reference responses. Mid-tier models, such as Llama2, Llama3, and Mistral benefit significantly from NMISS, highlighting their ability to provide richer contextual information. This combined approach offers new insights into the reduction and more accurate assessment of hallucinations in LLMs, with applications in real-world healthcare tasks and other domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05357v2">Model-GLUE: Democratized LLM Scaling for A Large Model Zoo in the Wild</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 24 pages, 4 figures, accepted to NeurIPS 2024 Datasets and Benchmarks Track
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) excel across tasks and specialized domains, scaling LLMs based on existing models has garnered significant attention, which faces the challenge of decreasing performance when combining disparate models. Various techniques have been proposed for the aggregation of pre-trained LLMs, including model merging, Mixture-of-Experts, and stacking. Despite their merits, a comprehensive comparison and synergistic application of them to a diverse model zoo is yet to be adequately addressed. In light of this research gap, this paper introduces Model-GLUE, a holistic LLM scaling guideline. First, our work starts with a benchmarking of existing LLM scaling techniques, especially selective merging, and variants of mixture. Utilizing the insights from the benchmark results, we formulate an optimal strategy for the selection and aggregation of a heterogeneous model zoo characterizing different architectures and initialization.Our methodology involves the clustering of mergeable models and optimal merging strategy selection, and the integration of clusters through a model mixture. Finally, evidenced by our experiments on a diverse Llama-2-based model zoo, Model-GLUE shows an average performance enhancement of 5.61%, achieved without additional training. Codes are available at: https://github.com/Model-GLUE/Model-GLUE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02785v2">Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 v2: Updated with changes from peer review rebuttal. v1: Version under peer review
    </div>
    <details class="paper-abstract">
      Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt. Source code and data: https://github.com/uiuc-focal-lab/stochastic-monkeys/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04110v1">Enhancing Mathematical Reasoning in LLMs with Background Operators</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      We propose utilizing background operators for mathematical reasoning in large language models (LLMs). To achieve this, we define a set of fundamental mathematical predicates as the basic building blocks. For each mathematical problem, we develop a Prolog solution that includes problem-specific predicates and intermediate predicates derived from these background operators, ensuring that each solution adheres to the defined operator set. We introduce the MATH-Prolog corpus, which is derived from the counting and probability categories of the MATH corpus. For efficient data augmentation, we apply K-fold cross-validated self-training. This method incrementally generates new Prolog solutions for each fold, incorporating those verified as correct into the training set throughout the model training process. Our experimental results demonstrate that 5-fold crossvalidated self-training effectively identifies new, accurate Prolog solutions, achieving an accuracy of 84.6% on the cross-validated set, and 84.8% on the test set during fine-tuning the Meta-Llama-3.1-8B-Instruct model. This approach successfully uncovers new solutions with fully computable inference steps for previously unseen problems. Additionally, incorporating the background mathematical predicates into the prompt enhances solution coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04093v1">Practical Considerations for Agentic LLM Systems</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 15 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      As the strength of Large Language Models (LLMs) has grown over recent years, so too has interest in their use as the underlying models for autonomous agents. Although LLMs demonstrate emergent abilities and broad expertise across natural language domains, their inherent unpredictability makes the implementation of LLM agents challenging, resulting in a gap between related research and the real-world implementation of such systems. To bridge this gap, this paper frames actionable insights and considerations from the research community in the context of established application paradigms to enable the construction and facilitate the informed deployment of robust LLM agents. Namely, we position relevant research findings into four broad categories--Planning, Memory, Tools, and Control Flow--based on common practices in application-focused literature and highlight practical considerations to make when designing agentic LLMs for real-world applications, such as handling stochasticity and managing resources efficiently. While we do not conduct empirical evaluations, we do provide the necessary background for discussing critical aspects of agentic LLM designs, both in academia and industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04090v1">LossAgent: Towards Any Optimization Objectives for Image Processing with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      We present the first loss agent, dubbed LossAgent, for low-level image processing tasks, e.g., image super-resolution and restoration, intending to achieve any customized optimization objectives of low-level image processing in different practical applications. Notably, not all optimization objectives, such as complex hand-crafted perceptual metrics, text description, and intricate human feedback, can be instantiated with existing low-level losses, e.g., MSE loss. which presents a crucial challenge in optimizing image processing networks in an end-to-end manner. To eliminate this, our LossAgent introduces the powerful large language model (LLM) as the loss agent, where the rich textual understanding of prior knowledge empowers the loss agent with the potential to understand complex optimization objectives, trajectory, and state feedback from external environments in the optimization process of the low-level image processing networks. In particular, we establish the loss repository by incorporating existing loss functions that support the end-to-end optimization for low-level image processing. Then, we design the optimization-oriented prompt engineering for the loss agent to actively and intelligently decide the compositional weights for each loss in the repository at each optimization interaction, thereby achieving the required optimization trajectory for any customized optimization objectives. Extensive experiments on three typical low-level image processing tasks and multiple optimization objectives have shown the effectiveness and applicability of our proposed LossAgent. Code and pre-trained models will be available at https://github.com/lbc12345/LossAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07122v2">SCAR: Sparse Conditioned Autoencoders for Concept Detection and Steering in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 Accepted at Socially Responsible Language Modelling Research (SoLaR) Workshop at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in generating human-like text, but their output may not be aligned with the user or even produce harmful content. This paper presents a novel approach to detect and steer concepts such as toxicity before generation. We introduce the Sparse Conditioned Autoencoder (SCAR), a single trained module that extends the otherwise untouched LLM. SCAR ensures full steerability, towards and away from concepts (e.g., toxic content), without compromising the quality of the model's text generation on standard evaluation benchmarks. We demonstrate the effective application of our approach through a variety of concepts, including toxicity, safety, and writing style alignment. As such, this work establishes a robust framework for controlling LLM generations, ensuring their ethical and safe deployment in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04036v1">SocialMind: LLM-based Proactive AR Social Assistive System with Human-like Perception for In-situ Live Interactions</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      Social interactions are fundamental to human life. The recent emergence of large language models (LLMs)-based virtual assistants has demonstrated their potential to revolutionize human interactions and lifestyles. However, existing assistive systems mainly provide reactive services to individual users, rather than offering in-situ assistance during live social interactions with conversational partners. In this study, we introduce SocialMind, the first LLM-based proactive AR social assistive system that provides users with in-situ social assistance. SocialMind employs human-like perception leveraging multi-modal sensors to extract both verbal and nonverbal cues, social factors, and implicit personas, incorporating these social cues into LLM reasoning for social suggestion generation. Additionally, SocialMind employs a multi-tier collaborative generation strategy and proactive update mechanism to display social suggestions on Augmented Reality (AR) glasses, ensuring that suggestions are timely provided to users without disrupting the natural flow of conversation. Evaluations on three public datasets and a user study with 20 participants show that SocialMind achieves 38.3% higher engagement compared to baselines, and 95% of participants are willing to use SocialMind in their live social interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03987v1">MTMT: Consolidating Multiple Thinking Modes to Form a Thought Tree for Strengthening LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown limitations in tasks requiring complex logical reasoning and multi-step problem-solving. To address these challenges, researchers have employed carefully designed prompts and flowcharts, simulating human cognitive processes to enhance LLM performance, such as the Chain of Thought approach. In this paper, we introduce MTMT (Multi-thinking Modes Tree), a novel method that interacts with LLMs to construct a thought tree, simulating various advanced cognitive processes, including but not limited to association, counterfactual thinking, task decomposition, and comparison. By breaking down the original complex task into simpler sub-questions, MTMT facilitates easier problem-solving for LLMs, enabling more effective utilization of the latent knowledge within LLMs. We evaluate the performance of MTMT under different parameter configurations, using GPT-4o mini as the base model. Our results demonstrate that integrating multiple modes of thinking significantly enhances the ability of LLMs to handle complex tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03905v1">Integrating Various Software Artifacts for Better LLM-based Bug Localization and Program Repair</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 22 pages, 11 images, 9 tables, Manuscript submitted to a journal (2024)
    </div>
    <details class="paper-abstract">
      LLMs have garnered considerable attention for their potential to streamline Automated Program Repair (APR). LLM-based approaches can either insert the correct code or directly generate patches when provided with buggy methods. However, most of LLM-based APR methods rely on a single type of software information, without fully leveraging different software artifacts. Despite this, many LLM-based approaches do not explore which specific types of information best assist in APR. Addressing this gap is crucial for advancing LLM-based APR techniques. We propose DEVLoRe to use issue content (description and message) and stack error traces to localize buggy methods, then rely on debug information in buggy methods and issue content and stack error to localize buggy lines and generate plausible patches which can pass all unit tests. The results show that while issue content is particularly effective in assisting LLMs with fault localization and program repair, different types of software artifacts complement each other. By incorporating different artifacts, DEVLoRe successfully locates 49.3% and 47.6% of single and non-single buggy methods and generates 56.0% and 14.5% plausible patches for the Defects4J v2.0 dataset, respectively. This outperforms current state-of-the-art APR methods. The source code and experimental results of this work for replication are available at https://github.com/XYZboom/DEVLoRe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10659v4">Network Formation and Dynamics Among Multi-LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      Social networks fundamentally shape human opinions, behaviors, and the dissemination of information. As large language models (LLMs) like GPT, Claude, and Llama increasingly integrate into social and professional settings, understanding their behavior in the context of social interactions and network formation becomes essential. This study develops a framework to systematically examine whether the network formation behaviors of multiple LLMs approximate certain aspects of human network dynamics. By simulating interactions among LLM agents across various model families, we observe that these models consistently exhibit key patterns associated with social network principles including preferential attachment, triadic closure, homophily, community structure, and the small-world phenomenon when forming networks. Moreover, LLMs adapt their network formation strategies based on each network's characteristics, reflecting the context-dependent nature of human behavior: in Facebook networks, they prioritize triadic closure and homophily, mirroring close-knit friendships; in phone networks, homophily and preferential attachment dominate, capturing personal and professional connections, while in employment networks, LLMs favor heterophily and high-degree connections, aligning with career advancement dynamics. These results open new avenues for using LLMs in network science research, with potential applications in agent-based modeling and synthetic network generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11266v4">VersaTune: An Efficient Data Composition Framework for Training Multi-Capability LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      Large-scale pretrained models, particularly Large Language Models (LLMs), have exhibited remarkable capabilities in handling multiple tasks across domains due to their emergent properties. These capabilities are further augmented during the Supervised Fine-Tuning (SFT) phase. Despite their potential, existing work mainly focuses on domain-specific enhancements during fine-tuning, the challenge of which lies in catastrophic forgetting of knowledge across other domains. In this study, we introduce VersaTune, a novel data composition framework designed for enhancing LLMs' overall multi-ability performances during training. We categorize knowledge into distinct domains including law, medicine, finance, science, code, etc. We begin with detecting the distribution of domain-specific knowledge within the base model, followed by the training data composition that aligns with the model's existing knowledge distribution. During the training process, domain weights are dynamically adjusted based on their learnable potential and forgetting degree. Experimental results demonstrate that VersaTune achieves significant improvements in multi-domain performance, with an 35.21% enhancement in comprehensive multi-domain tasks. Additionally, in scenarios where specific domain optimization is required, VersaTune reduces the degradation of performance in other domains by 38.77%, without compromising the target domain's training efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03815v1">Synergizing LLMs and Knowledge Graphs: A Novel Approach to Software Repository-Related Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 Submitted to ACM Transactions on Software Engineering and Methodology for review
    </div>
    <details class="paper-abstract">
      Software repositories contain valuable information for gaining insights into their development process. However, extracting insights from these repository data is time-consuming and requires technical expertise. While software engineering chatbots have been developed to facilitate natural language interactions with repositories, they struggle with understanding natural language and accurately retrieving relevant data. This study aims to improve the accuracy of LLM-based chatbots in answering repository-related questions by augmenting them with knowledge graphs. We achieve this in a two-step approach; (1) constructing a knowledge graph from the repository data and (2) synergizing the knowledge graph with LLM to allow for the natural language questions and answers. We curated a set of 20 questions with different complexities and evaluated our approach on five popular open-source projects. Our approach achieved an accuracy of 65%. We further investigated the limitations and identified six key issues, with the majority relating to the reasoning capability of the LLM. We experimented with a few-shot chain-of-thought prompting to determine if it could enhance our approach. This technique improved the overall accuracy to 84%. Our findings demonstrate the synergy between LLMs and knowledge graphs as a viable solution for making repository data accessible to both technical and non-technical stakeholders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03809v1">EditScout: Locating Forged Regions from Diffusion-based Edited Images with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      Image editing technologies are tools used to transform, adjust, remove, or otherwise alter images. Recent research has significantly improved the capabilities of image editing tools, enabling the creation of photorealistic and semantically informed forged regions that are nearly indistinguishable from authentic imagery, presenting new challenges in digital forensics and media credibility. While current image forensic techniques are adept at localizing forged regions produced by traditional image manipulation methods, current capabilities struggle to localize regions created by diffusion-based techniques. To bridge this gap, we present a novel framework that integrates a multimodal Large Language Model (LLM) for enhanced reasoning capabilities to localize tampered regions in images produced by diffusion model-based editing methods. By leveraging the contextual and semantic strengths of LLMs, our framework achieves promising results on MagicBrush, AutoSplice, and PerfBrush (novel diffusion-based dataset) datasets, outperforming previous approaches in mIoU and F1-score metrics. Notably, our method excels on the PerfBrush dataset, a self-constructed test set featuring previously unseen types of edits. Here, where traditional methods typically falter, achieving markedly low scores, our approach demonstrates promising performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11681v4">Designing LLM Chains by Adapting Techniques from Crowdsourcing Workflows</a></div>
    <div class="paper-meta">
      📅 2024-12-05
    </div>
    <details class="paper-abstract">
      LLM chains enable complex tasks by decomposing work into a sequence of subtasks. Similarly, the more established techniques of crowdsourcing workflows decompose complex tasks into smaller tasks for human crowdworkers. Chains address LLM errors analogously to the way crowdsourcing workflows address human error. To characterize opportunities for LLM chaining, we survey 107 papers across the crowdsourcing and chaining literature to construct a design space for chain development. The design space covers a designer's objectives and the tactics used to build workflows. We then surface strategies that mediate how workflows use tactics to achieve objectives. To explore how techniques from crowdsourcing may apply to chaining, we adapt crowdsourcing workflows to implement LLM chains across three case studies: creating a taxonomy, shortening text, and writing a short story. From the design space and our case studies, we identify takeaways for effective chain design and raise implications for future research and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04125v2">Exploring RAG-based Vulnerability Augmentation with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-05
      | 💬 13 pages, 6 figures, 5 tables, 3 prompt templates, 1 algorithm
    </div>
    <details class="paper-abstract">
      Detecting vulnerabilities is vital for software security, yet deep learning-based vulnerability detectors (DLVD) face a data shortage, which limits their effectiveness. Data augmentation can potentially alleviate the data shortage, but augmenting vulnerable code is challenging and requires a generative solution that maintains vulnerability. Previous works have only focused on generating samples that contain single statements or specific types of vulnerabilities. Recently, large language models (LLMs) have been used to solve various code generation and comprehension tasks with inspiring results, especially when fused with retrieval augmented generation (RAG). Therefore, we propose VulScribeR, a novel LLM-based solution that leverages carefully curated prompt templates to augment vulnerable datasets. More specifically, we explore three strategies to augment both single and multi-statement vulnerabilities, with LLMs, namely Mutation, Injection, and Extension. Our extensive evaluation across three vulnerability datasets and DLVD models, using two LLMs, show that our approach beats two SOTA methods Vulgen and VGX, and Random Oversampling (ROS) by 27.48%, 27.93%, and 15.41% in f1-score with 5K generated vulnerable samples on average, and 53.84%, 54.10%, 69.90%, and 40.93% with 15K generated vulnerable samples. Our approach demonstrates its feasibility for large-scale data augmentation by generating 1K samples at as cheap as US$ 1.88.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03213v1">ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been widely deployed in a variety of applications, and the context length is rapidly increasing to handle tasks such as long-document QA and complex logical reasoning. However, long context poses significant challenges for inference efficiency, including high memory costs of key-value (KV) cache and increased latency due to extensive memory accesses. Recent works have proposed compressing KV cache to approximate computation, but these methods either evict tokens permanently, never recalling them for later inference, or recall previous tokens at the granularity of pages divided by textual positions. Both approaches degrade the model accuracy and output quality. To achieve efficient and accurate recallable KV cache compression, we introduce ClusterKV, which recalls tokens at the granularity of semantic clusters. We design and implement efficient algorithms and systems for clustering, selection, indexing and caching. Experiment results show that ClusterKV attains negligible accuracy loss across various tasks with 32k context lengths, using only a 1k to 2k KV cache budget, and achieves up to a 2$\times$ speedup in latency and a 2.5$\times$ improvement in decoding throughput. Compared to SoTA recallable KV compression methods, ClusterKV demonstrates higher model accuracy and output quality, while maintaining or exceeding inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16926v2">Pyramid Vector Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Recent works on compression of large language models (LLM) using quantization considered reparameterizing the architecture such that weights are distributed on the sphere. This demonstratively improves the ability to quantize by increasing the mathematical notion of coherence, resulting in fewer weight outliers without affecting the network output. In this work, we aim to further exploit this spherical geometry of the weights when performing quantization by considering Pyramid Vector Quantization (PVQ) for large language models. Arranging points evenly on the sphere is notoriously difficult, especially in high dimensions, and in case approximate solutions exists, representing points explicitly in a codebook is typically not feasible due to its additional memory cost. Instead, PVQ uses a fixed integer lattice on the sphere by projecting points onto the 1-sphere, which allows for efficient encoding and decoding without requiring an explicit codebook in memory. To obtain a practical algorithm, we propose to combine PVQ with scale quantization for which we derive theoretically optimal quantizations, under empirically verified assumptions. Further, we extend pyramid vector quantization to use Hessian information to minimize quantization error under expected feature activations, instead of only relying on weight magnitudes. Experimentally, we achieves state-of-the-art quantization performance with pareto-optimal trade-off between performance and bits per weight and bits per activation, compared to compared methods. On weight-only, we find that we can quantize a Llama-3 70B model to 3.25 bits per weight and retain 98\% accuracy on downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01289v2">Enhancing Perception Capabilities of Multimodal LLMs with Training-Free Fusion</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Multimodal LLMs (MLLMs) equip language models with visual capabilities by aligning vision encoders with language models. Existing methods to enhance the visual perception of MLLMs often involve designing more powerful vision encoders, which requires exploring a vast design space and re-aligning each potential encoder with the language model, resulting in prohibitively high training costs. In this paper, we introduce VisionFuse, a novel integration framework that efficiently utilizes multiple vision encoders from off-the-shelf MLLMs to enhance visual perception without requiring additional training. Our approach is motivated by the observation that different MLLMs tend to focus on distinct regions given the same query and image. Moreover, we find that the feature distributions of vision encoders within an MLLM family, a group of MLLMs sharing the same pretrained LLM, are highly aligned. Building on these insights, VisionFuse enriches the visual context by concatenating the tokens generated by the vision encoders of selected MLLMs within a family. By merging the parameters of language models from these MLLMs, VisionFuse allows a single language model to align with various vision encoders, significantly reducing deployment overhead. We conduct comprehensive evaluations across multiple multimodal benchmarks using various MLLM combinations, demonstrating substantial improvements in multimodal tasks. Notably, when integrating MiniGemini-8B and SLIME-8B, VisionFuse achieves an average performance increase of over 4%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03123v1">Robust Multi-bit Text Watermark with LLM-based Paraphrasers</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      We propose an imperceptible multi-bit text watermark embedded by paraphrasing with LLMs. We fine-tune a pair of LLM paraphrasers that are designed to behave differently so that their paraphrasing difference reflected in the text semantics can be identified by a trained decoder. To embed our multi-bit watermark, we use two paraphrasers alternatively to encode the pre-defined binary code at the sentence level. Then we use a text classifier as the decoder to decode each bit of the watermark. Through extensive experiments, we show that our watermarks can achieve over 99.99\% detection AUC with small (1.1B) text paraphrasers while keeping the semantic information of the original sentence. More importantly, our pipeline is robust under word substitution and sentence paraphrasing perturbations and generalizes well to out-of-distributional data. We also show the stealthiness of our watermark with LLM-based evaluation. We open-source the code: https://github.com/xiaojunxu/multi-bit-text-watermark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04509v1">Pragmatic Metacognitive Prompting Improves LLM Performance on Sarcasm Detection</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Accepted at COLING 2024, CHum Workshop
    </div>
    <details class="paper-abstract">
      Sarcasm detection is a significant challenge in sentiment analysis due to the nuanced and context-dependent nature of verbiage. We introduce Pragmatic Metacognitive Prompting (PMP) to improve the performance of Large Language Models (LLMs) in sarcasm detection, which leverages principles from pragmatics and reflection helping LLMs interpret implied meanings, consider contextual cues, and reflect on discrepancies to identify sarcasm. Using state-of-the-art LLMs such as LLaMA-3-8B, GPT-4o, and Claude 3.5 Sonnet, PMP achieves state-of-the-art performance on GPT-4o on MUStARD and SemEval2018. This study demonstrates that integrating pragmatic reasoning and metacognitive strategies into prompting significantly enhances LLMs' ability to detect sarcasm, offering a promising direction for future research in sentiment analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00721v2">A Comparative Study of LLM-based ASR and Whisper in Low Resource and Code Switching Scenario</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 This work hasn't been finished yet
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have showcased exceptional performance across diverse NLP tasks, and their integration with speech encoder is rapidly emerging as a dominant trend in the Automatic Speech Recognition (ASR) field. Previous works mainly concentrated on leveraging LLMs for speech recognition in English and Chinese. However, their potential for addressing speech recognition challenges in low resource settings remains underexplored. Hence, in this work, we aim to explore the capability of LLMs in low resource ASR and Mandarin-English code switching ASR. We also evaluate and compare the recognition performance of LLM-based ASR systems against Whisper model. Extensive experiments demonstrate that LLM-based ASR yields a relative gain of 12.8\% over the Whisper model in low resource ASR while Whisper performs better in Mandarin-English code switching ASR. We hope that this study could shed light on ASR for low resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03605v1">CBEval: A framework for evaluating and interpreting cognitive biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Rapid advancements in Large Language models (LLMs) has significantly enhanced their reasoning capabilities. Despite improved performance on benchmarks, LLMs exhibit notable gaps in their cognitive processes. Additionally, as reflections of human-generated data, these models have the potential to inherit cognitive biases, raising concerns about their reasoning and decision making capabilities. In this paper we present a framework to interpret, understand and provide insights into a host of cognitive biases in LLMs. Conducting our research on frontier language models we're able to elucidate reasoning limitations and biases, and provide reasoning behind these biases by constructing influence graphs that identify phrases and words most responsible for biases manifested in LLMs. We further investigate biases such as round number bias and cognitive bias barrier revealed when noting framing effect in language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01130v2">Enhancing Function-Calling Capabilities in LLMs: Strategies for Prompt Formats, Data Integration, and Multilingual Translation</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced autonomous agents, particularly in zero-shot tool usage, also known as function calling. This research delves into enhancing the function-calling capabilities of LLMs by exploring different approaches, including prompt formats for integrating function descriptions, blending function-calling and instruction-following data, introducing a novel Decision Token for conditional prompts, leveraging chain-of-thought reasoning, and overcoming multilingual challenges with a translation pipeline. Our key findings and contributions are as follows: (1) Instruction-following data improves both function-calling accuracy and relevance detection. (2) The use of the newly proposed Decision Token, combined with synthetic non-function-call data, enhances relevance detection. (3) A tailored translation pipeline effectively overcomes multilingual limitations, demonstrating significant improvements in Traditional Chinese. These insights highlight the potential for improved function-calling capabilities and multilingual applications in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02956v1">Curriculum-style Data Augmentation for LLM-based Metaphor Detection</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Recently, utilizing large language models (LLMs) for metaphor detection has achieved promising results. However, these methods heavily rely on the capabilities of closed-source LLMs, which come with relatively high inference costs and latency. To address this, we propose a method for metaphor detection by fine-tuning open-source LLMs, effectively reducing inference costs and latency with a single inference step. Furthermore, metaphor detection suffers from a severe data scarcity problem, which hinders effective fine-tuning of LLMs. To tackle this, we introduce Curriculum-style Data Augmentation (CDA). Specifically, before fine-tuning, we evaluate the training data to identify correctly predicted instances for fine-tuning, while incorrectly predicted instances are used as seed data for data augmentation. This approach enables the model to quickly learn simpler knowledge and progressively acquire more complex knowledge, thereby improving performance incrementally. Experimental results demonstrate that our method achieves state-of-the-art performance across all baselines. Additionally, we provide detailed ablation studies to validate the effectiveness of CDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17422v2">Multimodal LLM Guided Exploration and Active Mapping using Fisher Information</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      We present an active mapping system that could plan for long-horizon exploration goals and short-term actions with a 3D Gaussian Splatting (3DGS) representation. Existing methods either did not take advantage of recent developments in multimodal Large Language Models (LLM) or did not consider challenges in localization uncertainty, which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based algorithm. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01109v2">Mixing It Up: The Cocktail Effect of Multi-Task Fine-Tuning on LLM Performance -- A Case Study in Finance</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in domain-specific contexts, including finance, has expanded rapidly. Domain-specific LLMs are typically evaluated based on their performance in various downstream tasks relevant to the domain. In this work, we present a detailed analysis of fine-tuning LLMs for such tasks. Somewhat counterintuitively, we find that in domain-specific cases, fine-tuning exclusively on the target task is not always the most effective strategy. Instead, multi-task finetuning - where models are trained on a cocktail of related tasks - can significantly enhance performance. We demonstrate how this approach enables a small model, such as Phi-3-Mini, to achieve state-of-the-art results, even surpassing the much larger GPT-4-o model on financial benchmarks. Our study involves a large-scale experiment, conducting over 200 training experiments using several widely adopted LLMs as baselines, and empirically confirms the benefits of multi-task fine-tuning. Additionally, we explore the use of general instruction data as a form of regularization, suggesting that it helps minimize performance degradation. We also investigate the inclusion of mathematical data, finding improvements in numerical reasoning that transfer effectively to financial tasks. Finally, we note that while fine-tuning for downstream tasks leads to targeted improvements in task performance, it does not necessarily result in broader gains in domain knowledge or complex domain reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19379v2">Marconi: Prefix Caching for the Era of Hybrid LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Hybrid models that combine the language modeling capabilities of Attention layers with the efficiency of Recurrent layers (e.g., State Space Models) have gained traction in practically supporting long contexts in Large Language Model serving. Yet, the unique properties of these models complicate the usage of complementary efficiency optimizations such as prefix caching that skip redundant computations across requests. Most notably, their use of in-place state updates for recurrent layers precludes rolling back cache entries for partial sequence overlaps, and instead mandates only exact-match cache hits; the effect is a deluge of (large) cache entries per sequence, most of which yield minimal reuse opportunities. We present Marconi, the first system that supports efficient prefix caching with Hybrid LLMs. Key to Marconi are its novel admission and eviction policies that more judiciously assess potential cache entries based not only on recency, but also on (1) forecasts of their reuse likelihood across a taxonomy of different hit scenarios, and (2) the compute savings that hits deliver relative to memory footprints. Across diverse workloads and Hybrid models, Marconi achieves up to 34.4$\times$ higher token hit rates (71.1% or 617 ms lower TTFT) compared to state-of-the-art prefix caching systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03516v1">You're (Not) My Type -- Can LLMs Generate Feedback of Specific Types for Introductory Programming Tasks?</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Accepted at Journal of Computer Assisted Learning (2024)
    </div>
    <details class="paper-abstract">
      Background: Feedback as one of the most influential factors for learning has been subject to a great body of research. It plays a key role in the development of educational technology systems and is traditionally rooted in deterministic feedback defined by experts and their experience. However, with the rise of generative AI and especially Large Language Models (LLMs), we expect feedback as part of learning systems to transform, especially for the context of programming. In the past, it was challenging to automate feedback for learners of programming. LLMs may create new possibilities to provide richer, and more individual feedback than ever before. Objectives: This paper aims to generate specific types of feedback for introductory programming tasks using LLMs. We revisit existing feedback taxonomies to capture the specifics of the generated feedback, such as randomness, uncertainty, and degrees of variation. Methods: We iteratively designed prompts for the generation of specific feedback types (as part of existing feedback taxonomies) in response to authentic student programs. We then evaluated the generated output and determined to what extent it reflected certain feedback types. Results and Conclusion: The present work provides a better understanding of different feedback dimensions and characteristics. The results have implications for future feedback research with regard to, for example, feedback effects and learners' informational needs. It further provides a basis for the development of new tools and learning systems for novice programmers including feedback generated by AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16860v2">Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 NeurIPS 2024 (Oral). Website at https://cambrian-mllm.github.io
    </div>
    <details class="paper-abstract">
      We introduce Cambrian-1, a family of multimodal LLMs (MLLMs) designed with a vision-centric approach. While stronger language models can enhance multimodal capabilities, the design choices for vision components are often insufficiently explored and disconnected from visual representation learning research. This gap hinders accurate sensory grounding in real-world scenarios. Our study uses LLMs and visual instruction tuning as an interface to evaluate various visual representations, offering new insights into different models and architectures -- self-supervised, strongly supervised, or combinations thereof -- based on experiments with over 20 vision encoders. We critically examine existing MLLM benchmarks, address the difficulties involved in consolidating and interpreting results from various tasks, and introduce a new vision-centric benchmark, CV-Bench. To further improve visual grounding, we propose the Spatial Vision Aggregator (SVA), a dynamic and spatially-aware connector that integrates high-resolution vision features with LLMs while reducing the number of tokens. Additionally, we discuss the curation of high-quality visual instruction-tuning data from publicly available sources, emphasizing the importance of data source balancing and distribution ratio. Collectively, Cambrian-1 not only achieves state-of-the-art performance but also serves as a comprehensive, open cookbook for instruction-tuned MLLMs. We provide model weights, code, supporting tools, datasets, and detailed instruction-tuning and evaluation recipes. We hope our release will inspire and accelerate advancements in multimodal systems and visual representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11376v2">Towards Time Series Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Oral Presentation at 2024 NeurIPS Workshop on Time Series in the Age of Large Models
    </div>
    <details class="paper-abstract">
      Multi-modal large language models (MLLMs) have enabled numerous advances in understanding and reasoning in domains like vision, but we have not yet seen this broad success for time-series. Although prior works on time-series MLLMs have shown promising performance in time-series forecasting, very few works show how an LLM could be used for time-series reasoning in natural language. We propose a novel multi-modal time-series LLM approach that learns generalizable information across various domains with powerful zero-shot performance. First, we train a lightweight time-series encoder on top of an LLM to directly extract time-series information. Then, we fine-tune our model with chain-of-thought augmented time-series tasks to encourage the model to generate reasoning paths. We show that our model learns a latent representation that reflects specific time-series features (e.g. slope, frequency), as well as outperforming GPT-4o on a set of zero-shot reasoning tasks on a variety of domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12071v2">Beyond the Comfort Zone: Emerging Solutions to Overcome Challenges in Integrating LLMs into Software Products</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 10 pages, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded into software products across diverse industries, enhancing user experiences, but at the same time introducing numerous challenges for developers. Unique characteristics of LLMs force developers, who are accustomed to traditional software development and evaluation, out of their comfort zones as the LLM components shatter standard assumptions about software systems. This study explores the emerging solutions that software developers are adopting to navigate the encountered challenges. Leveraging a mixed-method research, including 26 interviews and a survey with 332 responses, the study identifies 19 emerging solutions regarding quality assurance that practitioners across several product teams at Microsoft are exploring. The findings provide valuable insights that can guide the development and evaluation of LLM-based products more broadly in the face of these challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02205v2">DataLab: A Unified Platform for LLM-Powered Business Intelligence</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Business intelligence (BI) transforms large volumes of data within modern organizations into actionable insights for informed decision-making. Recently, large language model (LLM)-based agents have streamlined the BI workflow by automatically performing task planning, reasoning, and actions in executable environments based on natural language (NL) queries. However, existing approaches primarily focus on individual BI tasks such as NL2SQL and NL2VIS. The fragmentation of tasks across different data roles and tools lead to inefficiencies and potential errors due to the iterative and collaborative nature of BI. In this paper, we introduce DataLab, a unified BI platform that integrates a one-stop LLM-based agent framework with an augmented computational notebook interface. DataLab supports a wide range of BI tasks for different data roles by seamlessly combining LLM assistance with user customization within a single environment. To achieve this unification, we design a domain knowledge incorporation module tailored for enterprise-specific BI tasks, an inter-agent communication mechanism to facilitate information sharing across the BI workflow, and a cell-based context management strategy to enhance context utilization efficiency in BI notebooks. Extensive experiments demonstrate that DataLab achieves state-of-the-art performance on various BI tasks across popular research benchmarks. Moreover, DataLab maintains high effectiveness and efficiency on real-world datasets from Tencent, achieving up to a 58.58% increase in accuracy and a 61.65% reduction in token cost on enterprise-specific BI tasks.
    </details>
</div>
