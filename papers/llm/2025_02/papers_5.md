# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14450v1">LLM4FaaS: No-Code Application Development using LLMs and FaaS</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are powerful tools that can generate code from natural language descriptions. While this theoretically enables non-technical users to develop their own applications, they typically lack the expertise to execute, deploy, and operate generated code. This poses a barrier for such users to leverage the power of LLMs for application development. In this paper, we propose leveraging the high levels of abstraction of the Function-as-a-Service (FaaS) paradigm to handle code execution and operation for non-technical users. FaaS offers function deployment without handling the underlying infrastructure, enabling users to execute LLM-generated code without concern for its operation and without requiring any technical expertise. We propose LLM4FaaS, a novel no-code application development approach that combines LLMs and FaaS platforms to enable non-technical users to build and run their own applications using only natural language descriptions. Specifically, LLM4FaaS takes user prompts, uses LLMs to generate function code based on those prompts, and deploys these functions through a FaaS platform that handles the application's operation. LLM4FaaS also leverages the FaaS infrastructure abstractions to reduce the task complexity for the LLM, improving result accuracy. We evaluate LLM4FaaS with a proof-of-concept implementation based on GPT-4o and an open-source FaaS platform, using real prompts from non-technical users. Our evaluation based on these real user prompts demonstrates the feasibility of our approach and shows that LLM4FaaS can reliably build and deploy code in 71.47% of cases, up from 43.48% in a baseline without FaaS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14445v1">PredictaBoard: Benchmarking LLM Score Predictability</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Despite possessing impressive skills, Large Language Models (LLMs) often fail unpredictably, demonstrating inconsistent success in even basic common sense reasoning tasks. This unpredictability poses a significant challenge to ensuring their safe deployment, as identifying and operating within a reliable "safe zone" is essential for mitigating risks. To address this, we present PredictaBoard, a novel collaborative benchmarking framework designed to evaluate the ability of score predictors (referred to as assessors) to anticipate LLM errors on specific task instances (i.e., prompts) from existing datasets. PredictaBoard evaluates pairs of LLMs and assessors by considering the rejection rate at different tolerance errors. As such, PredictaBoard stimulates research into developing better assessors and making LLMs more predictable, not only with a higher average performance. We conduct illustrative experiments using baseline assessors and state-of-the-art LLMs. PredictaBoard highlights the critical need to evaluate predictability alongside performance, paving the way for safer AI systems where errors are not only minimised but also anticipated and effectively mitigated. Code for our benchmark can be found at https://github.com/Kinds-of-Intelligence-CFI/PredictaBoard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12769v2">How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14389v1">Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessment</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Argument mining algorithms analyze the argumentative structure of essays, making them a valuable tool for enhancing education by providing targeted feedback on the students' argumentation skills. While current methods often use encoder or encoder-decoder deep learning architectures, decoder-only models remain largely unexplored, offering a promising research direction. This paper proposes leveraging open-source, small Large Language Models (LLMs) for argument mining through few-shot prompting and fine-tuning. These models' small size and open-source nature ensure accessibility, privacy, and computational efficiency, enabling schools and educators to adopt and deploy them locally. Specifically, we perform three tasks: segmentation of student essays into arguments, classification of the arguments by type, and assessment of their quality. We empirically evaluate the models on the Feedback Prize - Predicting Effective Arguments dataset of grade 6-12 students essays and demonstrate how fine-tuned small LLMs outperform baseline methods in segmenting the essays and determining the argument types while few-shot prompting yields comparable performance to that of the baselines in assessing quality. This work highlights the educational potential of small, open-source LLMs to provide real-time, personalized feedback, enhancing independent learning and writing skills while ensuring low computational cost and privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14359v1">Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      We examine three evaluation paradigms: large question-answering benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14328v1">SolSearch: An LLM-Driven Framework for Efficient SAT-Solving Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The Satisfiability (SAT) problem is a core challenge with significant applications in software engineering, including automated testing, configuration management, and program verification. This paper presents SolSearch, a novel framework that harnesses large language models (LLMs) to discover and optimize SAT-solving strategies automatically. Leveraging a curriculum-based, trial-and-error process, SolSearch enables the LLM to iteratively modify and generate SAT solver code, thereby improving solving efficiency and performance. This automated SAT-solving paradigm has the advantage of being plug-and-play, allowing integration with any SAT solver and accelerating the development or design process of new SAT solvers (new methods). Our preliminary experimental results are encouraging by demonstrating that the LLM-powered paradigm improves state-of-the-art SAT solvers on general SAT benchmarks and significantly enhances the performance of the widely used Z3 solver (11\% on PAR-2 score). These results highlight the potential for using LLM-driven methods to advance solver adaptability and effectiveness in real-world software engineering challenges. Future research directions are discussed to further refine and validate this approach, offering a promising avenue for integrating AI with traditional software engineering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14321v1">Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated remarkable capabilities in reasoning, planning, and decision-making. Building upon these strengths, researchers have begun incorporating LLMs into multi-agent systems (MAS), where agents collaborate or compete through natural language interactions to tackle tasks beyond the scope of single-agent setups. In this survey, we present a communication-centric perspective on LLM-based multi-agent systems, examining key system-level features such as architecture design and communication goals, as well as internal mechanisms like communication strategies, paradigms, objects and content. We illustrate how these communication elements interplay to enable collective intelligence and flexible collaboration. Furthermore, we discuss prominent challenges, including scalability, security, and multimodal integration, and propose directions for future work to advance research in this emerging domain. Ultimately, this survey serves as a catalyst for further innovation, fostering more robust, scalable, and intelligent multi-agent systems across diverse application domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14305v1">Efficient AI in Practice: Training and Deployment of Efficient LLMs for Industry Applications</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across a wide range of industrial applications, from search and recommendations to generative tasks. Although scaling laws indicate that larger models generally yield better generalization and performance, their substantial computational requirements often render them impractical for many real-world scenarios at scale. In this paper, we present methods and insights for training small language models (SLMs) that deliver high performance and efficiency in deployment. We focus on two key techniques: (1) knowledge distillation and (2) model compression via quantization and pruning. These approaches enable SLMs to retain much of the quality of their larger counterparts while significantly reducing training, serving costs, and latency. We detail the impact of these techniques on a variety of use cases at a large professional social network platform and share deployment lessons - including hardware optimization strategies that enhance speed and throughput for both predictive and reasoning-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13497v2">Towards Geo-Culturally Grounded LLM Generations</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have been demonstrated to have gaps in diverse, cultural knowledge across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on the ability of LLMs to display familiarity with a diverse range of national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on a series of cultural familiarity benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., the norms, artifacts, and institutions of national cultures), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models, while failing to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional knowledge about a culture and open-ended cultural fluency when it comes to evaluating the cultural familiarity of generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05040v2">SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Our code, data, and model will be released at https://github.com/InternLM/SWE-Fixer
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency across a variety of complex tasks. One significant application of LLMs is in tackling software engineering challenges, particularly in resolving real-world tasks on GitHub by fixing code based on the issues reported by the users. However, many current approaches rely on proprietary LLMs, which limits reproducibility, accessibility, and transparency. The critical components of LLMs for addressing software engineering issues and how their capabilities can be effectively enhanced remain unclear. To address these challenges, we introduce SWE-Fixer, a novel open-source framework designed to effectively and efficiently resolve GitHub issues. SWE-Fixer comprises two essential modules: a code file retrieval module and a code editing module. The retrieval module employs BM25 along with a lightweight model to achieve coarse-to-fine file retrieval. Subsequently, the code editing module utilizes the other model to generate patches for the identified files. To mitigate the lack of publicly available datasets, we compile an extensive dataset that includes 110K GitHub issues along with their corresponding patches and train the two models of SWE-Fixer separately. We assess our approach on the SWE-Bench Lite and Verified benchmarks, achieving state-of-the-art performance among open-source models with scores of 24.7% and 32.8%, respectively. Additionally, our approach requires only two model calls per instance, making it significantly more efficient than existing methods. These results highlight the effectiveness of SWE-Fixer in real-world code-fixing scenarios. We will make our model, dataset, and code publicly available at https://github.com/InternLM/SWE-Fixer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16406v4">SpinQuant: LLM quantization with learned rotations</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) techniques applied to weights, activations, and the KV cache greatly reduce memory usage, latency, and power consumption of Large Language Models (LLMs), but may lead to large quantization errors when outliers are present. Rotating activation or weight matrices helps remove outliers and benefits quantization. In this work, we identify a collection of applicable rotation parameterizations that lead to identical outputs in full-precision Transformer architectures while enhancing quantization accuracy. In addition, we find that some random rotations lead to much better quantization than others, with an up to 13 points difference in downstream zero-shot reasoning performance. As a result, we propose SpinQuant, a novel approach that incorporates learned rotation matrices for optimal quantized network accuracy. With 4-bit quantization of weight, activation, and KV-cache, SpinQuant narrows the accuracy gap on zero-shot reasoning tasks with full precision to merely 2.9 points on the LLaMA-2 7B model, surpassing LLM-QAT by 19.1 points and SmoothQuant by 25.0 points. Furthermore, SpinQuant also outperforms concurrent work QuaRot, which applies random rotations to remove outliers. In particular, for LLaMA-3 8B models that are hard to quantize, SpinQuant reduces the gap to full precision by up to 45.1% relative to QuaRot. Code is available at https://github.com/facebookresearch/SpinQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14276v1">STeCa: Step-level Trajectory Calibration for LLM Agent Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents have shown promise in tackling complex tasks by interacting dynamically with the environment. Existing work primarily focuses on behavior cloning from expert demonstrations and preference learning through exploratory trajectory sampling. However, these methods often struggle in long-horizon tasks, where suboptimal actions accumulate step by step, causing agents to deviate from correct task trajectories. To address this, we highlight the importance of timely calibration and the need to automatically construct calibration trajectories for training agents. We propose Step-Level Trajectory Calibration (STeCa), a novel framework for LLM agent learning. Specifically, STeCa identifies suboptimal actions through a step-level reward comparison during exploration. It constructs calibrated trajectories using LLM-driven reflection, enabling agents to learn from improved decision-making processes. These calibrated trajectories, together with successful trajectory data, are utilized for reinforced training. Extensive experiments demonstrate that STeCa significantly outperforms existing methods. Further analysis highlights that step-level calibration enables agents to complete tasks with greater robustness. Our code and data are available at https://github.com/WangHanLinHenry/STeCa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14273v1">LLM-EvRep: Learning an LLM-Compatible Event Representation Using a Self-Supervised Framework</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 6 pages, 2 figures,Companion Proceedings of the ACM Web Conference 2025 (WWW Companion '25)
    </div>
    <details class="paper-abstract">
      Recent advancements in event-based recognition have demonstrated significant promise, yet most existing approaches rely on extensive training, limiting their adaptability for efficient processing of event-driven visual content. Meanwhile, large language models (LLMs) have exhibited remarkable zero-shot capabilities across diverse domains, but their application to event-based visual recognition remains largely unexplored. To bridge this gap, we propose \textbf{LLM-EvGen}, an event representation generator that produces LLM-compatible event representations \textbf{LLM-EvRep}, thereby enhancing the performance of LLMs on event recognition tasks. The generator is trained using a self-supervised framework, aligning the generated representations with semantic consistency and structural fidelity. Comprehensive experiments were conducted on three datasets: N-ImageNet, N-Caltech101, and N-MNIST. The results demonstrate that our method, \textbf{LLM-EvRep}, outperforms the event-to-video method, E2VID, by 15.93\%, 0.82\%, and 50.21\%, respectively, in recognition tasks when evaluated using GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14271v1">PaperHelper: Knowledge-Based LLM QA Paper Reading Assistant</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      In the paper, we introduce a paper reading assistant, PaperHelper, a potent tool designed to enhance the capabilities of researchers in efficiently browsing and understanding scientific literature. Utilizing the Retrieval-Augmented Generation (RAG) framework, PaperHelper effectively minimizes hallucinations commonly encountered in large language models (LLMs), optimizing the extraction of accurate, high-quality knowledge. The implementation of advanced technologies such as RAFT and RAG Fusion significantly boosts the performance, accuracy, and reliability of the LLMs-based literature review process. Additionally, PaperHelper features a user-friendly interface that facilitates the batch downloading of documents and uses the Mermaid format to illustrate structural relationships between documents. Experimental results demonstrate that PaperHelper, based on a fine-tuned GPT-4 API, achieves an F1 Score of 60.04, with a latency of only 5.8 seconds, outperforming the basic RAG model by 7\% in F1 Score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04323v2">SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Accepted by The ACM Web Conference 2025 (WWW'25)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are gaining increasing popularity across a wide range of web applications, it is of great importance to optimize service-level objectives (SLOs) for LLM inference services to enhance user satisfaction and improve the competitiveness of cloud vendors. In this paper, we observe that adjusting the parameters of LLM inference engines can improve service performance, and the optimal parameter configurations of different services are different. Therefore, we propose SCOOT, an automatic performance tuning system to optimize SLOs for each LLM inference service by tuning the parameters of the inference engine. SCOOT jointly exploits single-objective and multiple-objective Bayesian optimization (BO) techniques to handle various optimization objectives via exploration and exploitation. Moreover, SCOOT prunes the search space with known constraints and adopts a random forest to learn hidden constraints during the tuning process to mitigate invalid exploration. To improve the tuning efficiency, SCOOT utilizes the parallel suggestion to accelerate the tuning process. Extensive experiments demonstrate that SCOOT considerably outperforms existing tuning techniques in SLO optimization while greatly improving the tuning efficiency. Moreover, SCOOT is universally applicable to various LLM inference engines including vLLM and TensorRT-LLM. Currently, SCOOT has already been implemented in the production environment at Ant Group.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14251v2">Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Post-training is essential for enabling large language models (LLMs) to follow human instructions. However, its effectiveness depends on high-quality instruction data, which is challenging to obtain in the real world due to privacy concerns, data scarcity, and high annotation costs. To fill this gap, inspired by the recent success of using LLMs to simulate human society, we propose MATRIX, a multi-agent simulator that automatically generates diverse text-based scenarios, capturing a wide range of real-world human needs in a realistic and scalable manner. Leveraging these outputs, we introduce a novel scenario-driven instruction generator MATRIX-Gen for controllable and highly realistic data synthesis. Extensive experiments demonstrate that our framework effectively generates both general and domain-specific data. On AlpacaEval 2 and Arena-Hard benchmarks, Llama-3-8B-Base, post-trained on datasets synthesized by MATRIX-Gen with just 20K instruction-response pairs, outperforms Meta's Llama-3-8B-Instruct model, which was trained on over 10M pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14219v1">Investigating the Impact of LLM Personality on Cognitive Bias Manifestation in Automated Decision-Making Tasks</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in decision-making, yet their susceptibility to cognitive biases remains a pressing challenge. This study explores how personality traits influence these biases and evaluates the effectiveness of mitigation strategies across various model architectures. Our findings identify six prevalent cognitive biases, while the sunk cost and group attribution biases exhibit minimal impact. Personality traits play a crucial role in either amplifying or reducing biases, significantly affecting how LLMs respond to debiasing techniques. Notably, Conscientiousness and Agreeableness may generally enhance the efficacy of bias mitigation strategies, suggesting that LLMs exhibiting these traits are more receptive to corrective measures. These findings address the importance of personality-driven bias dynamics and highlight the need for targeted mitigation approaches to improve fairness and reliability in AI-assisted decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14215v1">Towards Secure Program Partitioning for Smart Contracts with LLM's In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Smart contracts are highly susceptible to manipulation attacks due to the leakage of sensitive information. Addressing manipulation vulnerabilities is particularly challenging because they stem from inherent data confidentiality issues rather than straightforward implementation bugs. To tackle this by preventing sensitive information leakage, we present PartitionGPT, the first LLM-driven approach that combines static analysis with the in-context learning capabilities of large language models (LLMs) to partition smart contracts into privileged and normal codebases, guided by a few annotated sensitive data variables. We evaluated PartitionGPT on 18 annotated smart contracts containing 99 sensitive functions. The results demonstrate that PartitionGPT successfully generates compilable, and verified partitions for 78% of the sensitive functions while reducing approximately 30% code compared to function-level partitioning approach. Furthermore, we evaluated PartitionGPT on nine real-world manipulation attacks that lead to a total loss of 25 million dollars, PartitionGPT effectively prevents eight cases, highlighting its potential for broad applicability and the necessity for secure program partitioning during smart contract development to diminish manipulation vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07355v2">Think Together and Work Better: Combining Humans' and LLMs' Think-Aloud Outcomes for Effective Text Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This study introduces \textbf{InteractEval}, a framework that integrates human expertise and Large Language Models (LLMs) using the Think-Aloud (TA) method to generate attributes for checklist-based text evaluation. By combining human flexibility and reasoning with LLM consistency, InteractEval outperforms traditional non-LLM-based and LLM-based baselines across four distinct dimensions, consisting of Coherence, Fluency, Consistency, and Relevance. The experiment also investigates the effectiveness of the TA method, showing that it promotes divergent thinking in both humans and LLMs, leading to the generation of a wider range of relevant attributes and enhance text evaluation performance. Comparative analysis reveals that humans excel at identifying attributes related to internal quality (Coherence and Fluency), but LLMs perform better at those attributes related to external alignment (Consistency and Relevance). Consequently, leveraging both humans and LLMs together produces the best evaluation outcomes. In other words, this study emphasizes the necessity of effectively combining humans and LLMs in an automated checklist-based text evaluation framework. The code is available at \textbf{\url{https://github.com/BBeeChu/InteractEval.git}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14202v1">Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The widespread adoption of conversational LLMs for software development has raised new security concerns regarding the safety of LLM-generated content. Our motivational study outlines ChatGPT's potential in volunteering context-specific information to the developers, promoting safe coding practices. Motivated by this finding, we conduct a study to evaluate the degree of security awareness exhibited by three prominent LLMs: Claude 3, GPT-4, and Llama 3. We prompt these LLMs with Stack Overflow questions that contain vulnerable code to evaluate whether they merely provide answers to the questions or if they also warn users about the insecure code, thereby demonstrating a degree of security awareness. Further, we assess whether LLM responses provide information about the causes, exploits, and the potential fixes of the vulnerability, to help raise users' awareness. Our findings show that all three models struggle to accurately detect and warn users about vulnerabilities, achieving a detection rate of only 12.6% to 40% across our datasets. We also observe that the LLMs tend to identify certain types of vulnerabilities related to sensitive information exposure and improper input neutralization much more frequently than other types, such as those involving external control of file names or paths. Furthermore, when LLMs do issue security warnings, they often provide more information on the causes, exploits, and fixes of vulnerabilities compared to Stack Overflow responses. Finally, we provide an in-depth discussion on the implications of our findings and present a CLI-based prompting tool that can be used to generate significantly more secure LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14192v1">NLP-AKG: Few-Shot Construction of NLP Academic Knowledge Graph Based on LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely applied in question answering over scientific research papers. To enhance the professionalism and accuracy of responses, many studies employ external knowledge augmentation. However, existing structures of external knowledge in scientific literature often focus solely on either paper entities or domain concepts, neglecting the intrinsic connections between papers through shared domain concepts. This results in less comprehensive and specific answers when addressing questions that combine papers and concepts. To address this, we propose a novel knowledge graph framework that captures deep conceptual relations between academic papers, constructing a relational network via intra-paper semantic elements and inter-paper citation relations. Using a few-shot knowledge graph construction method based on LLM, we develop NLP-AKG, an academic knowledge graph for the NLP domain, by extracting 620,353 entities and 2,271,584 relations from 60,826 papers in ACL Anthology. Based on this, we propose a 'sub-graph community summary' method and validate its effectiveness on three NLP scientific literature question answering datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14182v1">Multi-Faceted Studies on Data Poisoning can Advance LLM Development</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The lifecycle of large language models (LLMs) is far more complex than that of traditional machine learning models, involving multiple training stages, diverse data sources, and varied inference methods. While prior research on data poisoning attacks has primarily focused on the safety vulnerabilities of LLMs, these attacks face significant challenges in practice. Secure data collection, rigorous data cleaning, and the multistage nature of LLM training make it difficult to inject poisoned data or reliably influence LLM behavior as intended. Given these challenges, this position paper proposes rethinking the role of data poisoning and argue that multi-faceted studies on data poisoning can advance LLM development. From a threat perspective, practical strategies for data poisoning attacks can help evaluate and address real safety risks to LLMs. From a trustworthiness perspective, data poisoning can be leveraged to build more robust LLMs by uncovering and mitigating hidden biases, harmful outputs, and hallucinations. Moreover, from a mechanism perspective, data poisoning can provide valuable insights into LLMs, particularly the interplay between data and model behavior, driving a deeper understanding of their underlying mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10490v3">Learning Dynamics of LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Learning dynamics, which describes how the learning of specific training examples influences the model's predictions on other examples, gives us a powerful tool for understanding the behavior of deep learning systems. We study the learning dynamics of large language models during different types of finetuning, by analyzing the step-wise decomposition of how influence accumulates among different potential responses. Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. In particular, we propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning, e.g., the model might use phrases or facts in the response for question B to answer question A, or the model might keep repeating similar simple phrases when generating responses. We also extend our framework and highlight a unique "squeezing effect" to explain a previously observed phenomenon in off-policy direct preference optimization (DPO), where running DPO for too long makes even the desired outputs less likely. This framework also provides insights into where the benefits of on-policy DPO and other variants come from. The analysis not only provides a novel perspective of understanding LLM's finetuning but also inspires a simple, effective method to improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15097v1">LUME: LLM Unlearning with Multitask Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Unlearning aims to remove copyrighted, sensitive, or private content from large language models (LLMs) without a full retraining. In this work, we develop a multi-task unlearning benchmark (LUME) which features three tasks: (1) unlearn synthetically generated creative short novels, (2) unlearn synthetic biographies with sensitive information, and (3) unlearn a collection of public biographies. We further release two fine-tuned LLMs of 1B and 7B parameter sizes as the target models. We conduct detailed evaluations of several recently proposed unlearning algorithms and present results on carefully crafted metrics to understand their behavior and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05315v2">Aligned at the Start: Conceptual Groupings in LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This paper shifts focus to the often-overlooked input embeddings - the initial representations fed into transformer blocks. Using fuzzy graph, k-nearest neighbor (k-NN), and community detection, we analyze embeddings from diverse LLMs, finding significant categorical community structure aligned with predefined concepts and categories aligned with humans. We observe these groupings exhibit within-cluster organization (such as hierarchies, topological ordering, etc.), hypothesizing a fundamental structure that precedes contextual processing. To further investigate the conceptual nature of these groupings, we explore cross-model alignments across different LLM categories within their input embeddings, observing a medium to high degree of alignment. Furthermore, provide evidence that manipulating these groupings can play a functional role in mitigating ethnicity bias in LLM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15090v1">Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) achieve impressive performance on some tasks, while exhibiting distinctly non-human-like behaviors on others. This raises the question of how well the LLM's learned representations align with human representations. In this work, we introduce a novel approach to the study of representation alignment: we adopt a method from research on activation steering to identify neurons responsible for specific concepts (e.g., 'cat') and then analyze the corresponding activation patterns. Our findings reveal that LLM representations closely align with human representations inferred from behavioral data. Notably, this alignment surpasses that of word embeddings, which have been center stage in prior work on human and model alignment. Additionally, our approach enables a more granular view of how LLMs represent concepts. Specifically, we show that LLMs organize concepts in a way that reflects hierarchical relationships interpretable to humans (e.g., 'animal'-'dog').
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04616v2">Can LLMs Improve Multimodal Fact-Checking by Asking Relevant Questions?</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Traditional fact-checking relies on humans to formulate relevant and targeted fact-checking questions (FCQs), search for evidence, and verify the factuality of claims. While Large Language Models (LLMs) have been commonly used to automate evidence retrieval and factuality verification at scale, their effectiveness for fact-checking is hindered by the absence of FCQ formulation. To bridge this gap, we seek to answer two research questions: (1) Can LLMs generate relevant FCQs? (2) Can LLM-generated FCQs improve multimodal fact-checking? We therefore introduce a framework LRQ-FACT for using LLMs to generate relevant FCQs to facilitate evidence retrieval and enhance fact-checking by probing information across multiple modalities. Through extensive experiments, we verify if LRQ-FACT can generate relevant FCQs of different types and if LRQ-FACT can consistently outperform baseline methods in multimodal fact-checking. Further analysis illustrates how each component in LRQ-FACT works toward improving the fact-checking performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09933v2">MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 32 pages, 11 figures. v2 updated the project page and dataset link
    </div>
    <details class="paper-abstract">
      Inductive Reasoning (IR), the ability to summarize rules from examples and apply on new ones, has long been viewed as a primal ability for general intelligence and widely studied by cognitive science and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually $<$10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations are mostly focused on classification (a very limited aspect of IR), and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context inductive reasoning benchmark that asks LLM to induce output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for inductive reasoning and many-shot ICL, including robustness against erroneous shots and the effect of Chain-of-Thought (CoT), and acquired insightful findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15018v1">Using tournaments to calculate AUROC for zero-shot classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models perform surprisingly well on many zero-shot classification tasks, but are difficult to fairly compare to supervised classifiers due to the lack of a modifiable decision boundary. In this work, we propose and evaluate a method that converts binary classification tasks into pairwise comparison tasks, obtaining relative rankings from LLMs. Repeated pairwise comparisons can be used to score instances using the Elo rating system (used in chess and other competitions), inducing a confidence ordering over instances in a dataset. We evaluate scheduling algorithms for their ability to minimize comparisons, and show that our proposed algorithm leads to improved classification performance, while also providing more information than traditional zero-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15009v1">Contextualizing Search Queries In-Context Learning for Conversational Rewriting with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Conversational query rewriting is crucial for effective conversational search, yet traditional supervised methods require substantial labeled data, which is scarce in low-resource settings. This paper introduces Prompt-Guided In-Context Learning, a novel approach that leverages the in-context learning capabilities of Large Language Models (LLMs) for few-shot conversational query rewriting. Our method employs carefully designed prompts, incorporating task descriptions, input/output format specifications, and a small set of illustrative examples, to guide pre-trained LLMs to generate context-independent queries without explicit fine-tuning. Extensive experiments on benchmark datasets, TREC and Taskmaster-1, demonstrate that our approach significantly outperforms strong baselines, including supervised models and contrastive co-training methods, across various evaluation metrics such as BLEU, ROUGE-L, Success Rate, and MRR. Ablation studies confirm the importance of in-context examples, and human evaluations further validate the superior fluency, relevance, and context utilization of our generated rewrites. The results highlight the potential of prompt-guided in-context learning as an efficient and effective paradigm for low-resource conversational query rewriting, reducing the reliance on extensive labeled data and complex training procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14461v2">From Human Annotation to LLMs: SILICON Annotation Workflow for Management Research</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Unstructured text data annotation and analysis are fundamental to management research, often relying on human annotators through crowdsourcing platforms. While Large Language Models (LLMs) promise to provide a cost-effective and efficient alternative to human annotation, there lacks a systematic workflow that evaluate when LLMs are suitable or how to proceed with LLM-based text annotation in a reproducible manner. This paper addresses this methodological gap by introducing the ``SILICON" (Systematic Inference with LLMs for Information Classification and Notation) workflow. The workflow integrates established principles of human annotation with systematic prompt optimization and model selection, addressing challenges such as developing robust annotation guidelines, establishing high-quality human baselines, optimizing prompts, and ensuring reproducibility across LLMs. We validate the SILICON workflow through seven case studies covering common management research tasks. Our findings highlight the importance of validating annotation guideline agreement, the superiority of expert-developed human baselines over crowdsourced ones, the iterative nature of prompt optimization, and the necessity of testing multiple LLMs. We also find that LLMs agree well with expert annotations in most cases but show low agreement in more complex multi-label classification tasks. Notably, we propose a regression-based methodology to empirically compare LLM outputs across prompts and models. Our workflow advances management research by establishing rigorous, transparent, and reproducible processes for LLM-based annotation. We provide practical guidance for researchers to effectively navigate the evolving landscape of generative AI tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03884v3">AlphaPO -- Reward shape matters for LLM alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Some popular examples of DAAs include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce \textbf{AlphaPO}, a new DAA method that leverages an $\alpha$-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7\% to 10\% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B while achieving 15\% to 50\% relative improvement over DPO on the same models. The analysis and results presented highlight the importance of the reward shape, and how one can systematically change it to affect training dynamics, as well as improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15835v1">Pragmatic Reasoning improves LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive potential in translating natural language (NL) instructions into program code. However, user instructions often contain inherent ambiguities, making it challenging for LLMs to generate code that accurately reflects the user's true intent. To address this challenge, researchers have proposed to produce multiple candidates of the program code and then rerank them to identify the best solution. In this paper, we propose CodeRSA, a novel code candidate reranking mechanism built upon the Rational Speech Act (RSA) framework, designed to guide LLMs toward more comprehensive pragmatic reasoning about user intent. We evaluate CodeRSA using one of the latest LLMs on a popular code generation dataset. Our experiment results show that CodeRSA consistently outperforms common baselines, surpasses the state-of-the-art approach in most cases, and demonstrates robust overall performance. These findings underscore the effectiveness of integrating pragmatic reasoning into code candidate reranking, offering a promising direction for enhancing code generation quality in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14866v1">LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Accepted by MLSys 2025. Code available at: https://github.com/mit-han-lab/omniserve
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable potential in processing long sequences, yet efficiently serving these long-context models remains challenging due to the quadratic computational complexity of attention in the prefilling stage and the large memory footprint of the KV cache in the decoding stage. To address these issues, we introduce LServe, an efficient system that accelerates long-sequence LLM serving via hybrid sparse attention. This method unifies different hardware-friendly, structured sparsity patterns for both prefilling and decoding attention into a single framework, where computations on less important tokens are skipped block-wise. LServe demonstrates the compatibility of static and dynamic sparsity in long-context LLM attention. This design enables multiplicative speedups by combining these optimizations. Specifically, we convert half of the attention heads to nearly free streaming heads in both the prefilling and decoding stages. Additionally, we find that only a constant number of KV pages is required to preserve long-context capabilities, irrespective of context length. We then design a hierarchical KV page selection policy that dynamically prunes KV pages based on query-centric similarity. On average, LServe accelerates LLM prefilling by up to 2.9x and decoding by 1.3-2.1x over vLLM, maintaining long-context accuracy. Code is released at https://github.com/mit-han-lab/omniserve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14860v1">Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 22 pages, 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often fail to ask effective questions under uncertainty, making them unreliable in domains where proactive information-gathering is essential for decisionmaking. We present ALFA, a framework that improves LLM question-asking by (i) decomposing the notion of a "good" question into a set of theory-grounded attributes (e.g., clarity, relevance), (ii) controllably synthesizing attribute-specific question variations, and (iii) aligning models via preference-based optimization to explicitly learn to ask better questions along these fine-grained attributes. Focusing on clinical reasoning as a case study, we introduce the MediQ-AskDocs dataset, composed of 17k real-world clinical interactions augmented with 80k attribute-specific preference pairs of follow-up questions, as well as a novel expert-annotated interactive healthcare QA task to evaluate question-asking abilities. Models aligned with ALFA reduce diagnostic errors by 56.6% on MediQ-AskDocs compared to SOTA instruction-tuned LLMs, with a question-level win-rate of 64.4% and strong generalizability. Our findings suggest that explicitly guiding question-asking with structured, fine-grained attributes offers a scalable path to improve LLMs, especially in expert application domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14847v1">Red-Teaming LLM Multi-Agent Systems via Communication Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Model-based Multi-Agent Systems (LLM-MAS) have revolutionized complex problem-solving capability by enabling sophisticated agent collaboration through message-based communications. While the communication framework is crucial for agent coordination, it also introduces a critical yet unexplored security vulnerability. In this work, we introduce Agent-in-the-Middle (AiTM), a novel attack that exploits the fundamental communication mechanisms in LLM-MAS by intercepting and manipulating inter-agent messages. Unlike existing attacks that compromise individual agents, AiTM demonstrates how an adversary can compromise entire multi-agent systems by only manipulating the messages passing between agents. To enable the attack under the challenges of limited control and role-restricted communication format, we develop an LLM-powered adversarial agent with a reflection mechanism that generates contextually-aware malicious instructions. Our comprehensive evaluation across various frameworks, communication structures, and real-world applications demonstrates that LLM-MAS is vulnerable to communication-based attacks, highlighting the need for robust security measures in multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14837v1">Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 16 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a latent vector. Compared to MLA, standard LLMs employing Multi-Head Attention (MHA) and its variants such as Grouped-Query Attention (GQA) exhibit significant cost disadvantages. Enabling well-trained LLMs (e.g., Llama) to rapidly adapt to MLA without pre-training from scratch is both meaningful and challenging. This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for partial-RoPE, we remove RoPE from dimensions of queries and keys that contribute less to the attention scores, for low-rank approximation, we introduce joint SVD approximations based on the pre-trained parameters of keys and values. These carefully designed strategies enable MHA2MLA to recover performance using only a small fraction (0.3% to 0.6%) of the data, significantly reducing inference costs while seamlessly integrating with compression techniques such as KV cache quantization. For example, the KV cache size of Llama2-7B is reduced by 92.19%, with only a 0.5% drop in LongBench performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07752v2">Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Designing efficient optimizers for large language models (LLMs) with low-memory requirements and fast convergence is an important and challenging problem. This paper makes a step towards the systematic design of such optimizers through the lens of structured Fisher information matrix (FIM) approximation. We show that many state-of-the-art efficient optimizers can be viewed as solutions to FIM approximation (under the Frobenius norm) with specific structural assumptions. Building on these insights, we propose two design recommendations of practical efficient optimizers for LLMs, involving the careful selection of structural assumptions to balance generality and efficiency, and enhancing memory efficiency of optimizers with general structures through a novel low-rank extension framework. We demonstrate how to use each design approach by deriving new memory-efficient optimizers: Row and Column Scaled SGD (RACS) and Adaptive low-dimensional subspace estimation (Alice). Experiments on LLaMA pre-training (up to 1B parameters) validate the effectiveness, showing faster and better convergence than existing memory-efficient baselines and Adam with little memory overhead. Notably, Alice achieves better than 2x faster convergence over Adam, while RACS delivers strong performance on the 1B model with SGD-like memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14830v1">Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      While large language models demonstrate remarkable capabilities at task-specific applications through fine-tuning, extending these benefits across diverse languages is essential for broad accessibility. However, effective cross-lingual transfer is hindered by LLM performance gaps across languages and the scarcity of fine-tuning data in many languages. Through analysis of LLM internal representations from over 1,000+ language pairs, we discover that middle layers exhibit the strongest potential for cross-lingual alignment. Building on this finding, we propose a middle-layer alignment objective integrated into task-specific training. Our experiments on slot filling, machine translation, and structured text generation show consistent improvements in cross-lingual transfer, especially to lower-resource languages. The method is robust to the choice of alignment languages and generalizes to languages unseen during alignment. Furthermore, we show that separately trained alignment modules can be merged with existing task-specific modules, improving cross-lingual capabilities without full re-training. Our code is publicly available (https://github.com/dannigt/mid-align).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14828v1">Fundamental Limitations in Defending LLM Finetuning APIs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      LLM developers have imposed technical interventions to prevent fine-tuning misuse attacks, attacks where adversaries evade safeguards by fine-tuning the model using a public API. Previous work has established several successful attacks against specific fine-tuning API defences. In this work, we show that defences of fine-tuning APIs that seek to detect individual harmful training or inference samples ('pointwise' detection) are fundamentally limited in their ability to prevent fine-tuning attacks. We construct 'pointwise-undetectable' attacks that repurpose entropy in benign model outputs (e.g. semantic or syntactic variations) to covertly transmit dangerous knowledge. Our attacks are composed solely of unsuspicious benign samples that can be collected from the model before fine-tuning, meaning training and inference samples are all individually benign and low-perplexity. We test our attacks against the OpenAI fine-tuning API, finding they succeed in eliciting answers to harmful multiple-choice questions, and that they evade an enhanced monitoring system we design that successfully detects other fine-tuning attacks. We encourage the community to develop defences that tackle the fundamental limitations we uncover in pointwise fine-tuning API defences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14008v1">MaskPrune: Mask-based LLM Pruning for Layer-wise Uniform Structures</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The remarkable performance of large language models (LLMs) in various language tasks has attracted considerable attention. However, the ever-increasing size of these models presents growing challenges for deployment and inference. Structured pruning, an effective model compression technique, is gaining increasing attention due to its ability to enhance inference efficiency. Nevertheless, most previous optimization-based structured pruning methods sacrifice the uniform structure across layers for greater flexibility to maintain performance. The heterogeneous structure hinders the effective utilization of off-the-shelf inference acceleration techniques and impedes efficient configuration for continued training. To address this issue, we propose a novel masking learning paradigm based on minimax optimization to obtain the uniform pruned structure by optimizing the masks under sparsity regularization. Extensive experimental results demonstrate that our method can maintain high performance while ensuring the uniformity of the pruned model structure, thereby outperforming existing SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13996v1">Beyond Single-Value Metrics: Evaluating and Enhancing LLM Unlearning with Cognitive Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Due to the widespread use of LLMs and the rising critical ethical and safety concerns, LLM unlearning methods have been developed to remove harmful knowledge and undesirable capabilities. In this context, evaluations are mostly based on single-value metrics such as QA accuracy. However, these metrics often fail to capture the nuanced retention of harmful knowledge components, making it difficult to assess the true effectiveness of unlearning. To address this issue, we propose UNCD (UNlearning evaluation via Cognitive Diagnosis), a novel framework that leverages Cognitive Diagnosis Modeling for fine-grained evaluation of LLM unlearning. Our dedicated benchmark, UNCD-Cyber, provides a detailed assessment of the removal of dangerous capabilities. Moreover, we introduce UNCD-Agent, which refines unlearning by diagnosing knowledge remnants and generating targeted unlearning data. Extensive experiments across eight unlearning methods and two base models demonstrate that UNCD not only enhances evaluation but also effectively facilitates the removal of harmful LLM abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13965v1">Autellix: An Efficient Serving Engine for LLM Agents as General Programs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) applications are evolving beyond simple chatbots into dynamic, general-purpose agentic programs, which scale LLM calls and output tokens to help AI agents reason, explore, and solve complex tasks. However, existing LLM serving systems ignore dependencies between programs and calls, missing significant opportunities for optimization. Our analysis reveals that programs submitted to LLM serving engines experience long cumulative wait times, primarily due to head-of-line blocking at both the individual LLM request and the program. To address this, we introduce Autellix, an LLM serving system that treats programs as first-class citizens to minimize their end-to-end latencies. Autellix intercepts LLM calls submitted by programs, enriching schedulers with program-level context. We propose two scheduling algorithms-for single-threaded and distributed programs-that preempt and prioritize LLM calls based on their programs' previously completed calls. Our evaluation demonstrates that across diverse LLMs and agentic workloads, Autellix improves throughput of programs by 4-15x at the same latency compared to state-of-the-art systems, such as vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02890v4">Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13913v1">How Do LLMs Perform Two-Hop Reasoning in Context?</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      "Socrates is human. All humans are mortal. Therefore, Socrates is mortal." This classical example demonstrates two-hop reasoning, where a conclusion logically follows from two connected premises. While transformer-based Large Language Models (LLMs) can make two-hop reasoning, they tend to collapse to random guessing when faced with distracting premises. To understand the underlying mechanism, we train a three-layer transformer on synthetic two-hop reasoning tasks. The training dynamics show two stages: a slow learning phase, where the 3-layer transformer performs random guessing like LLMs, followed by an abrupt phase transitions, where the 3-layer transformer suddenly reaches $100%$ accuracy. Through reverse engineering, we explain the inner mechanisms for how models learn to randomly guess between distractions initially, and how they learn to ignore distractions eventually. We further propose a three-parameter model that supports the causal claims for the mechanisms to the training dynamics of the transformer. Finally, experiments on LLMs suggest that the discovered mechanisms generalize across scales. Our methodologies provide new perspectives for scientific understandings of LLMs and our findings provide new insights into how reasoning emerges during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13908v1">Judging the Judges: A Collection of LLM-Generated Relevance Judgements</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Using Large Language Models (LLMs) for relevance assessments offers promising opportunities to improve Information Retrieval (IR), Natural Language Processing (NLP), and related fields. Indeed, LLMs hold the promise of allowing IR experimenters to build evaluation collections with a fraction of the manual human labor currently required. This could help with fresh topics on which there is still limited knowledge and could mitigate the challenges of evaluating ranking systems in low-resource scenarios, where it is challenging to find human annotators. Given the fast-paced recent developments in the domain, many questions concerning LLMs as assessors are yet to be answered. Among the aspects that require further investigation, we can list the impact of various components in a relevance judgment generation pipeline, such as the prompt used or the LLM chosen. This paper benchmarks and reports on the results of a large-scale automatic relevance judgment evaluation, the LLMJudge challenge at SIGIR 2024, where different relevance assessment approaches were proposed. In detail, we release and benchmark 42 LLM-generated labels of the TREC 2023 Deep Learning track relevance judgments produced by eight international teams who participated in the challenge. Given their diverse nature, these automatically generated relevance judgments can help the community not only investigate systematic biases caused by LLMs but also explore the effectiveness of ensemble models, analyze the trade-offs between different models and human assessors, and advance methodologies for improving automated evaluation techniques. The released resource is available at the following link: https://llm4eval.github.io/LLMJudge-benchmark/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13897v1">DataSciBench: An LLM Agent Benchmark for Data Science</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 40 pages, 7 figures, 6 tables
    </div>
    <details class="paper-abstract">
      This paper presents DataSciBench, a comprehensive benchmark for evaluating Large Language Model (LLM) capabilities in data science. Recent related benchmarks have primarily focused on single tasks, easily obtainable ground truth, and straightforward evaluation metrics, which limits the scope of tasks that can be evaluated. In contrast, DataSciBench is constructed based on a more comprehensive and curated collection of natural and challenging prompts for uncertain ground truth and evaluation metrics. We develop a semi-automated pipeline for generating ground truth (GT) and validating evaluation metrics. This pipeline utilizes and implements an LLM-based self-consistency and human verification strategy to produce accurate GT by leveraging collected prompts, predefined task types, and aggregate functions (metrics). Furthermore, we propose an innovative Task - Function - Code (TFC) framework to assess each code execution outcome based on precisely defined metrics and programmatic rules. Our experimental framework involves testing 6 API-based models, 8 open-source general models, and 9 open-source code generation models using the diverse set of prompts we have gathered. This approach aims to provide a more comprehensive and rigorous evaluation of LLMs in data science, revealing their strengths and weaknesses. Experimental results demonstrate that API-based models outperform open-sourced models on all metrics and Deepseek-Coder-33B-Instruct achieves the highest score among open-sourced models. We release all code and data at https://github.com/THUDM/DataSciBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13870v1">SPEX: Scaling Feature Interaction Explanations for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized machine learning due to their ability to capture complex interactions between input features. Popular post-hoc explanation methods like SHAP provide marginal feature attributions, while their extensions to interaction importances only scale to small input lengths ($\approx 20$). We propose Spectral Explainer (SPEX), a model-agnostic interaction attribution algorithm that efficiently scales to large input lengths ($\approx 1000)$. SPEX exploits underlying natural sparsity among interactions -- common in real-world data -- and applies a sparse Fourier transform using a channel decoding algorithm to efficiently identify important interactions. We perform experiments across three difficult long-context datasets that require LLMs to utilize interactions between inputs to complete the task. For large inputs, SPEX outperforms marginal attribution methods by up to 20% in terms of faithfully reconstructing LLM outputs. Further, SPEX successfully identifies key features and interactions that strongly influence model output. For one of our datasets, HotpotQA, SPEX provides interactions that align with human annotations. Finally, we use our model-agnostic approach to generate explanations to demonstrate abstract reasoning in closed-source LLMs (GPT-4o mini) and compositional reasoning in vision-language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13845v1">Enhancing LLM-Based Recommendations Through Personalized Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 7 pages, under review
    </div>
    <details class="paper-abstract">
      Current recommendation systems powered by large language models (LLMs) often underutilize their reasoning capabilities due to a lack of explicit logical structuring. To address this limitation, we introduce CoT-Rec, a framework that integrates Chain-of-Thought (CoT) reasoning into LLM-driven recommendations by incorporating two crucial processes: user preference analysis and item perception evaluation. CoT-Rec operates in two key phases: (1) personalized data extraction, where user preferences and item perceptions are identified, and (2) personalized data application, where this information is leveraged to refine recommendations. Our experimental analysis demonstrates that CoT-Rec improves recommendation accuracy by making better use of LLMs' reasoning potential. The implementation is publicly available at https://anonymous.4open.science/r/CoT-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13843v1">Enhancing Cross-Domain Recommendations with Memory-Optimized LLM-Based User Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 6 pages, under review
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based user agents have emerged as a powerful tool for improving recommender systems by simulating user interactions. However, existing methods struggle with cross-domain scenarios due to inefficient memory structures, leading to irrelevant information retention and failure to account for social influence factors such as popularity. To address these limitations, we introduce AgentCF++, a novel framework featuring a dual-layer memory architecture and a two-step fusion mechanism to filter domain-specific preferences effectively. Additionally, we propose interest groups with shared memory, allowing the model to capture the impact of popularity trends on users with similar interests. Through extensive experiments on multiple cross-domain datasets, AgentCF++ demonstrates superior performance over baseline models, highlighting its effectiveness in refining user behavior simulation for recommender systems. Our code is available at https://anonymous.4open.science/r/AgentCF-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13834v1">Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 Published as a conference paper at ICLR 2025. Code is available at https://github.com/Lizn-zn/NeqLIPS/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~1). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13794v1">LESA: Learnable LLM Layer Scaling-Up</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) from scratch requires immense computational resources, making it prohibitively expensive. Model scaling-up offers a promising solution by leveraging the parameters of smaller models to create larger ones. However, existing depth scaling-up methods rely on empirical heuristic rules for layer duplication, which result in poorer initialization and slower convergence during continual pre-training. We propose \textbf{LESA}, a novel learnable method for depth scaling-up. By concatenating parameters from each layer and applying Singular Value Decomposition, we uncover latent patterns between layers, suggesting that inter-layer parameters can be learned. LESA uses a neural network to predict the parameters inserted between adjacent layers, enabling better initialization and faster training. Experiments show that LESA outperforms existing baselines, achieving superior performance with less than half the computational cost during continual pre-training. Extensive analyses demonstrate its effectiveness across different model sizes and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13791v1">From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in working environments for a wide range of tasks, excelling at solving individual problems in isolation. However, are they also able to effectively collaborate over long-term interactions? To investigate this, we introduce MemoryCode, a synthetic multi-session dataset designed to test LLMs' ability to track and execute simple coding instructions amid irrelevant information, simulating a realistic setting. While all the models we tested handle isolated instructions well, even the performance of state-of-the-art models like GPT-4o deteriorates when instructions are spread across sessions. Our analysis suggests this is due to their failure to retrieve and integrate information over long instruction chains. Our results highlight a fundamental limitation of current LLMs, restricting their ability to collaborate effectively in long interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13783v1">Generative Large Recommendation Models: Emerging Trends in LLMs for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 This paper has been accepted for the tutorial track at WWW 2025
    </div>
    <details class="paper-abstract">
      In the era of information overload, recommendation systems play a pivotal role in filtering data and delivering personalized content. Recent advancements in feature interaction and user behavior modeling have significantly enhanced the recall and ranking processes of these systems. With the rise of large language models (LLMs), new opportunities have emerged to further improve recommendation systems. This tutorial explores two primary approaches for integrating LLMs: LLMs-enhanced recommendations, which leverage the reasoning capabilities of general LLMs, and generative large recommendation models, which focus on scaling and sophistication. While the former has been extensively covered in existing literature, the latter remains underexplored. This tutorial aims to fill this gap by providing a comprehensive overview of generative large recommendation models, including their recent advancements, challenges, and potential research directions. Key topics include data quality, scaling laws, user behavior mining, and efficiency in training and inference. By engaging with this tutorial, participants will gain insights into the latest developments and future opportunities in the field, aiding both academic research and practical applications. The timely nature of this exploration supports the rapid evolution of recommendation systems, offering valuable guidance for researchers and practitioners alike.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13691v1">Is This Collection Worth My LLM's Time? Automatically Measuring Information Potential in Text Corpora</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) converge towards similar capabilities, the key to advancing their performance lies in identifying and incorporating valuable new information sources. However, evaluating which text collections are worth the substantial investment required for digitization, preprocessing, and integration into LLM systems remains a significant challenge. We present a novel approach to this challenge: an automated pipeline that evaluates the potential information gain from text collections without requiring model training or fine-tuning. Our method generates multiple choice questions (MCQs) from texts and measures an LLM's performance both with and without access to the source material. The performance gap between these conditions serves as a proxy for the collection's information potential. We validate our approach using three strategically selected datasets: EPFL PhD manuscripts (likely containing novel specialized knowledge), Wikipedia articles (presumably part of training data), and a synthetic baseline dataset. Our results demonstrate that this method effectively identifies collections containing valuable novel information, providing a practical tool for prioritizing data acquisition and integration efforts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13681v1">An LLM-based Agent for Reliable Docker Environment Configuration</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Environment configuration is a critical yet time-consuming step in software development, especially when dealing with unfamiliar code repositories. While Large Language Models (LLMs) demonstrate the potential to accomplish software engineering tasks, existing methods for environment configuration often rely on manual efforts or fragile scripts, leading to inefficiencies and unreliable outcomes. We introduce Repo2Run, the first LLM-based agent designed to fully automate environment configuration and generate executable Dockerfiles for arbitrary Python repositories. We address two major challenges: (1) enabling the LLM agent to configure environments within isolated Docker containers, and (2) ensuring the successful configuration process is recorded and accurately transferred to a Dockerfile without error. To achieve this, we propose atomic configuration synthesis, featuring a dual-environment architecture (internal and external environment) with a rollback mechanism to prevent environment "pollution" from failed commands, guaranteeing atomic execution (execute fully or not at all) and a Dockerfile generator to transfer successful configuration steps into runnable Dockerfiles. We evaluate Repo2Run~on our proposed benchmark of 420 recent Python repositories with unit tests, where it achieves an 86.0% success rate, outperforming the best baseline by 63.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17003v4">Safety Layers in Aligned Large Language Models: The Key to LLM Security</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Aligned LLMs are secure, capable of recognizing and refusing to answer malicious questions. However, the role of internal parameters in maintaining such security is not well understood yet, further these models can be vulnerable to security degradation when subjected to fine-tuning attacks. To address these challenges, our work uncovers the mechanism behind security in aligned LLMs at the parameter level, identifying a small set of contiguous layers in the middle of the model that are crucial for distinguishing malicious queries from normal ones, referred to as ``safety layers". We first confirm the existence of these safety layers by analyzing variations in input vectors within the model's internal layers. Additionally, we leverage the over-rejection phenomenon and parameters scaling analysis to precisely locate the safety layers. Building on these findings, we propose a novel fine-tuning approach, Safely Partial-Parameter Fine-Tuning (SPPFT), that fixes the gradient of the safety layers during fine-tuning to address the security degradation. Our experiments demonstrate that the proposed approach can significantly preserve LLM security while maintaining performance and reducing computational resources compared to full fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13648v1">Reliability Across Parametric and External Knowledge: Understanding Knowledge Handling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 under-review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enhance their problem-solving capability by leveraging both parametric and external knowledge. Beyond leveraging external knowledge to improve response accuracy, they require key capabilities for reliable knowledge-handling: resolving conflicts between knowledge sources, avoiding distraction from uninformative external knowledge, and abstaining when sufficient knowledge is unavailable. Prior studies have examined these scenarios in isolation or with limited scope. To systematically evaluate these capabilities, we introduce a comprehensive framework for analyzing knowledge-handling based on two key dimensions: the presence of parametric knowledge and the informativeness of external knowledge. Through analysis, we identify biases in knowledge utilization and examine how the ability to handle one scenario impacts performance in others. Furthermore, we demonstrate that training on data constructed based on the knowledge-handling scenarios improves LLMs' reliability in integrating and utilizing knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13640v1">Qorgau: Evaluating LLM Safety in Kazakh-Russian Bilingual Contexts</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to have the potential to generate harmful content, posing risks to users. While significant progress has been made in developing taxonomies for LLM risks and safety evaluation prompts, most studies have focused on monolingual contexts, primarily in English. However, language- and region-specific risks in bilingual contexts are often overlooked, and core findings can diverge from those in monolingual settings. In this paper, we introduce Qorgau, a novel dataset specifically designed for safety evaluation in Kazakh and Russian, reflecting the unique bilingual context in Kazakhstan, where both Kazakh (a low-resource language) and Russian (a high-resource language) are spoken. Experiments with both multilingual and language-specific LLMs reveal notable differences in safety performance, emphasizing the need for tailored, region-specific datasets to ensure the responsible and safe deployment of LLMs in countries like Kazakhstan. Warning: this paper contains example data that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13632v1">Concept Layers: Enhancing Interpretability and Intervenability via LLM Conceptualization</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The opaque nature of Large Language Models (LLMs) has led to significant research efforts aimed at enhancing their interpretability, primarily through post-hoc methods. More recent in-hoc approaches, such as Concept Bottleneck Models (CBMs), offer both interpretability and intervenability by incorporating explicit concept representations. However, these methods suffer from key limitations, including reliance on labeled concept datasets and significant architectural modifications that challenges re-integration into existing system pipelines. In this work, we introduce a new methodology for incorporating interpretability and intervenability into an existing model by integrating Concept Layers (CLs) into its architecture. Our approach projects the model's internal vector representations into a conceptual, explainable vector space before reconstructing and feeding them back into the model. Furthermore, we eliminate the need for a human-selected concept set by algorithmically searching an ontology for a set of concepts that can be either task-specific or task-agnostic. We evaluate CLs across multiple tasks, demonstrating that they maintain the original model's performance and agreement while enabling meaningful interventions. Additionally, we present a proof of concept showcasing an intervenability interface, allowing users to adjust model behavior dynamically, such as mitigating biases during inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22118v2">The Impact of Inference Acceleration on Bias of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Last few years have seen unprecedented advances in capabilities of Large Language Models (LLMs). These advancements promise to benefit a vast array of application domains. However, due to their immense size, performing inference with LLMs is both costly and slow. Consequently, a plethora of recent work has proposed strategies to enhance inference efficiency, e.g., quantization, pruning, and caching. These acceleration strategies reduce the inference cost and latency, often by several factors, while maintaining much of the predictive performance measured via common benchmarks. In this work, we explore another critical aspect of LLM performance: demographic bias in model generations due to inference acceleration optimizations. Using a wide range of metrics, we probe bias in model outputs from a number of angles. Analysis of outputs before and after inference acceleration shows significant change in bias. Worryingly, these bias effects are complex and unpredictable. A combination of an acceleration strategy and bias type may show little bias change in one model but may lead to a large effect in another. Our results highlight a need for in-depth and case-by-case evaluation of model bias after it has been modified to accelerate inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13606v1">LaVCa: LLM-assisted Visual Cortex Captioning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 33 pages
    </div>
    <details class="paper-abstract">
      Understanding the property of neural populations (or voxels) in the human brain can advance our comprehension of human perceptual and cognitive processing capabilities and contribute to developing brain-inspired computer models. Recent encoding models using deep neural networks (DNNs) have successfully predicted voxel-wise activity. However, interpreting the properties that explain voxel responses remains challenging because of the black-box nature of DNNs. As a solution, we propose LLM-assisted Visual Cortex Captioning (LaVCa), a data-driven approach that uses large language models (LLMs) to generate natural-language captions for images to which voxels are selective. By applying LaVCa for image-evoked brain activity, we demonstrate that LaVCa generates captions that describe voxel selectivity more accurately than the previously proposed method. Furthermore, the captions generated by LaVCa quantitatively capture more detailed properties than the existing method at both the inter-voxel and intra-voxel levels. Furthermore, a more detailed analysis of the voxel-specific properties generated by LaVCa reveals fine-grained functional differentiation within regions of interest (ROIs) in the visual cortex and voxels that simultaneously represent multiple distinct concepts. These findings offer profound insights into human visual representations by assigning detailed captions throughout the visual cortex while highlighting the potential of LLM-based methods in understanding brain representations. Please check out our webpage at https://sites.google.com/view/lavca-llm/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13603v1">Efficient Safety Retrofitting Against Jailbreaking for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) is an efficient alignment technique that steers LLMs towards preferable outputs by training on preference data, bypassing the need for explicit reward models. Its simplicity enables easy adaptation to various domains and safety requirements. This paper examines DPO's effectiveness in model safety against jailbreaking attacks while minimizing data requirements and training costs. We introduce Egida, a dataset expanded from multiple sources, which includes 27 different safety topics and 18 different attack styles, complemented with synthetic and human labels. This data is used to boost the safety of state-of-the-art LLMs (Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack styles. In addition to safety evaluations, we assess their post-alignment performance degradation in general purpose tasks, and their tendency to over refusal. Following the proposed methodology, trained models reduce their Attack Success Rate by 10%-30%, using small training efforts (2,000 samples) with low computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned models generalize to unseen topics and attack styles, with the most successful attack style reaching a success rate around 5%. Size and family are found to strongly influence model malleability towards safety, pointing at the importance of pre-training choices. To validate our findings, a large independent assessment of human preference agreement with Llama-Guard-3-8B is conducted by the authors and the associated dataset Egida-HSafe is released. Overall, this study illustrates how affordable and accessible it is to enhance LLM safety using DPO while outlining its current limitations. All datasets and models are released to enable reproducibility and further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13577v1">Unraveling the Localized Latents: Learning Stratified Manifold Structures in LLM Embedding Space with Sparse Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      However, real-world data often exhibit complex local structures that can be challenging for single-model approaches with a smooth global manifold in the embedding space to unravel. In this work, we conjecture that in the latent space of these large language models, the embeddings live in a local manifold structure with different dimensions depending on the perplexities and domains of the input data, commonly referred to as a Stratified Manifold structure, which in combination form a structured space known as a Stratified Space. To investigate the validity of this structural claim, we propose an analysis framework based on a Mixture-of-Experts (MoE) model where each expert is implemented with a simple dictionary learning algorithm at varying sparsity levels. By incorporating an attention-based soft-gating network, we verify that our model learns specialized sub-manifolds for an ensemble of input data sources, reflecting the semantic stratification in LLM embedding space. We further analyze the intrinsic dimensions of these stratified sub-manifolds and present extensive statistics on expert assignments, gating entropy, and inter-expert distances. Our experimental results demonstrate that our method not only validates the claim of a stratified manifold structure in the LLM embedding space, but also provides interpretable clusters that align with the intrinsic semantic variations of the input data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13542v1">Activation-aware Probe-Query: Effective Key-Value Retrieval for Long-Context LLMs Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have showcased exceptional performance in long-context tasks, while facing significant inference efficiency challenges with limited GPU memory. Existing solutions first proposed the sliding-window approach to accumulate a set of historical \textbf{key-value} (KV) pairs for reuse, then further improvements selectively retain its subsets at each step. However, due to the sparse attention distribution across a long context, it is hard to identify and recall relevant KV pairs, as the attention is distracted by massive candidate pairs. Additionally, we found it promising to select representative tokens as probe-Query in each sliding window to effectively represent the entire context, which is an approach overlooked by existing methods. Thus, we propose \textbf{ActQKV}, a training-free, \textbf{Act}ivation-aware approach that dynamically determines probe-\textbf{Q}uery and leverages it to retrieve the relevant \textbf{KV} pairs for inference. Specifically, ActQKV monitors a token-level indicator, Activation Bias, within each context window, enabling the proper construction of probe-Query for retrieval at pre-filling stage. To accurately recall the relevant KV pairs and minimize the irrelevant ones, we design a dynamic KV cut-off mechanism guided by information density across layers at the decoding stage. Experiments on the Long-Bench and $\infty$ Benchmarks demonstrate its state-of-the-art performance with competitive inference quality and resource efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08904v2">MIH-TCCT: Mitigating Inconsistent Hallucinations in LLMs via Event-Driven Text-Code Cyclic Training</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Recent methodologies utilizing synthetic datasets have aimed to address inconsistent hallucinations in large language models (LLMs); however,these approaches are primarily tailored to specific tasks, limiting their generalizability. Inspired by the strong performance of code-trained models in logic-intensive domains, we propose a novel framework that leverages event-based text to generate corresponding code and employs cyclic training to transfer the logical consistency of code to natural language effectively. Our method significantly reduces inconsistent hallucinations across three leading LLMs and two categories of natural language tasks while maintaining overall performance. This framework effectively alleviates hallucinations without necessitating adaptation to downstream tasks, demonstrating generality and providing new perspectives to tackle the challenge of inconsistent hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13502v1">PLDR-LLMs Learn A Generalizable Tensor Operator That Can Replace Its Own Deep Neural Net At Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 15 pages, 1 figure, 12 tables
    </div>
    <details class="paper-abstract">
      We show that Large Language Model from Power Law Decoder Representations (PLDR-LLM) is a foundational model whose deductive outputs are invariant tensors up to a small perturbation. PLDR-LLM learns a singularity condition for the deductive outputs that enable the once-inferred energy-curvature tensor $\mathbf{G}_{LM}$ to replace the deep neural network of power law graph attention (PLGA) generating the deductive outputs at inference. We demonstrate that a cache for $\mathbf{G}_{LM}$ (G-cache) and KV-cache can be implemented in a straightforward manner to improve the inference time. The invariance and generalizable nature of deductive outputs is at a very high fidelity where deductive outputs have same RMSE and determinant values up to 15 decimal places after caching, and zero-shot benchmark scores remain unchanged. Ablation studies show that learned deductive outputs have distinct loss and accuracy characteristics from models pretrained with transferred, randomly initialized or identity tensors as a constant tensor operator and an LLM with scaled-dot product attention (SDPA) is a special case of PLDR-LLM where $\mathbf{G}_{LM}$ is predefined as identity. The observed invariance characteristic introduces a novel asymmetry between training and inference phases with caching. We outline observed common characteristics of the deductive outputs for the learned singularity condition. We provide an implementation of a training and inference framework for PLDR-LLM with KV-cache and G-cache.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11506v2">Glimpse: Enabling White-Box Methods to Use Proprietary Models for Zero-Shot LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 ICLR 2025 camera version (10 pages, 9 figures, 9 tables)
    </div>
    <details class="paper-abstract">
      Advanced large language models (LLMs) can generate text almost indistinguishable from human-written text, highlighting the importance of LLM-generated text detection. However, current zero-shot techniques face challenges as white-box methods are restricted to use weaker open-source LLMs, and black-box methods are limited by partial observation from stronger proprietary LLMs. It seems impossible to enable white-box methods to use proprietary models because API-level access to the models neither provides full predictive distributions nor inner embeddings. To traverse the divide, we propose **Glimpse**, a probability distribution estimation approach, predicting the full distributions from partial observations. Despite the simplicity of Glimpse, we successfully extend white-box methods like Entropy, Rank, Log-Rank, and Fast-DetectGPT to latest proprietary models. Experiments show that Glimpse with Fast-DetectGPT and GPT-3.5 achieves an average AUROC of about 0.95 in five latest source models, improving the score by 51% relative to the remaining space of the open source baseline. It demonstrates that the latest LLMs can effectively detect their own outputs, suggesting that advanced LLMs may be the best shield against themselves. We release our code and data at https://github.com/baoguangsheng/glimpse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13499v1">Hidden Darkness in LLM-Generated Designs: Exploring Dark Patterns in Ecommerce Web Components Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Recent work has highlighted the risks of LLM-generated content for a wide range of harmful behaviors, including incorrect and harmful code. In this work, we extend this by studying whether LLM-generated web design contains dark patterns. This work evaluated designs of ecommerce web components generated by four popular LLMs: Claude, GPT, Gemini, and Llama. We tested 13 commonly used ecommerce components (e.g., search, product reviews) and used them as prompts to generate a total of 312 components across all models. Over one-third of generated components contain at least one dark pattern. The majority of dark pattern strategies involve hiding crucial information, limiting users' actions, and manipulating them into making decisions through a sense of urgency. Dark patterns are also more frequently produced in components that are related to company interests. These findings highlight the need for interventions to prevent dark patterns during front-end code generation with LLMs and emphasize the importance of expanding ethical design education to a broader audience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13497v1">Towards Geo-Culturally Grounded LLM Generations</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have been demonstrated to have gaps in diverse, cultural knowledge across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on the ability of LLMs to display familiarity with a diverse range of national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on a series of cultural familiarity benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., the norms, artifacts, and institutions of national cultures), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models, while failing to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional knowledge about a culture and open-ended cultural fluency when it comes to evaluating the cultural familiarity of generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11149v2">Large Language-Geometry Model: When LLM meets Equivariance</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Accurately predicting 3D structures and dynamics of physical systems is crucial in scientific applications. Existing approaches that rely on geometric Graph Neural Networks (GNNs) effectively enforce $\mathrm{E}(3)$-equivariance, but they often fall in leveraging extensive broader information. While direct application of Large Language Models (LLMs) can incorporate external knowledge, they lack the capability for spatial reasoning with guaranteed equivariance. In this paper, we propose EquiLLM, a novel framework for representing 3D physical systems that seamlessly integrates E(3)-equivariance with LLM capabilities. Specifically, EquiLLM comprises four key components: geometry-aware prompting, an equivariant encoder, an LLM, and an equivariant adaptor. Essentially, the LLM guided by the instructive prompt serves as a sophisticated invariant feature processor, while 3D directional information is exclusively handled by the equivariant encoder and adaptor modules. Experimental results demonstrate that EquiLLM delivers significant improvements over previous methods across molecular dynamics simulation, human motion simulation, and antibody design, highlighting its promising generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09403v2">Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The rapid advancement of scientific progress requires innovative tools that can accelerate knowledge discovery. Although recent AI methods, particularly large language models (LLMs), have shown promise in tasks such as hypothesis generation and experimental design, they fall short of replicating the collaborative nature of real-world scientific practices, where diverse experts work together in teams to tackle complex problems. To address the limitations, we propose an LLM-based multi-agent system, i.e., Virtual Scientists (VirSci), designed to mimic the teamwork inherent in scientific research. VirSci organizes a team of agents to collaboratively generate, evaluate, and refine research ideas. Through comprehensive experiments, we demonstrate that this multi-agent approach outperforms the state-of-the-art method in producing novel scientific ideas. We further investigate the collaboration mechanisms that contribute to its tendency to produce ideas with higher novelty, offering valuable insights to guide future research and illuminating pathways toward building a robust system for autonomous scientific discovery. The code is available at https://github.com/open-sciencelab/Virtual-Scientists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09282v2">CRVQ: Channel-Relaxed Vector Quantization for Extreme Compression of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 7 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Powerful large language models (LLMs) are increasingly expected to be deployed with lower computational costs, enabling their capabilities on resource-constrained devices. Post-training quantization (PTQ) has emerged as a star approach to achieve this ambition, with best methods compressing weights to less than 2 bit on average. In this paper, we propose Channel-Relaxed Vector Quantization (CRVQ), a novel technique that significantly improves the performance of PTQ baselines at the cost of only minimal additional bits. This state-of-the-art extreme compression method achieves its results through two key innovations: (1) carefully selecting and reordering a very small subset of critical weight channels, and (2) leveraging extended codebooks to relax the constraint of critical channels. With our method, we demonstrate a 38.9\% improvement over the current strongest sub-2-bit PTQ baseline, enabling nearer lossless 1-bit compression. Furthermore, our approach offers flexible customization of quantization bit-width and performance, providing a wider range of deployment options for diverse hardware platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13442v1">TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now achieve near-human performance on standard math word problem benchmarks (e.g., GSM8K), yet their true reasoning ability remains disputed. A key concern is that models often produce confident, yet unfounded, answers to unanswerable problems. We introduce TreeCut, a synthetic dataset that systematically generates infinite unanswerable math word problems and their answerable counterparts, by representing each question as a tree and removing chosen necessary conditions. Experiments show TreeCut effectively induce hallucinations in large language models, including GPT-4o and o3-mini, with rates of 61% and 42% in their respective worst-case scenarios. Further analysis highlights that deeper or more complex trees, composite item names, and removing necessary condition near the middle of a path all increase the likelihood of hallucinations, underscoring the persistent challenges LLMs face in identifying unanswerable math problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.04047v3">AutoParLLM: GNN-guided Context Generation for Zero-Shot Code Parallelization using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      In-Context Learning (ICL) has been shown to be a powerful technique to augment the capabilities of LLMs for a diverse range of tasks. This work proposes \ourtool, a novel way to generate context using guidance from graph neural networks (GNNs) to generate efficient parallel codes. We evaluate \ourtool \xspace{} on $12$ applications from two well-known benchmark suites of parallel codes: NAS Parallel Benchmark and Rodinia Benchmark. Our results show that \ourtool \xspace{} improves the state-of-the-art LLMs (e.g., GPT-4) by 19.9\% in NAS and 6.48\% in Rodinia benchmark in terms of CodeBERTScore for the task of parallel code generation. Moreover, \ourtool \xspace{} improves the ability of the most powerful LLM to date, GPT-4, by achieving $\approx$17\% (on NAS benchmark) and $\approx$16\% (on Rodinia benchmark) better speedup. In addition, we propose \ourscore \xspace{} for evaluating the quality of the parallel code and show its effectiveness in evaluating parallel codes. \ourtool \xspace is available at https://github.com/quazirafi/AutoParLLM.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13417v1">RLTHF: Targeted Human Feedback for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) to align with user preferences is challenging due to the high cost of quality human annotations in Reinforcement Learning from Human Feedback (RLHF) and the generalizability limitations of AI Feedback. To address these challenges, we propose RLTHF, a human-AI hybrid framework that combines LLM-based initial alignment with selective human annotations to achieve full-human annotation alignment with minimal effort. RLTHF identifies hard-to-annotate samples mislabeled by LLMs using a reward model's reward distribution and iteratively enhances alignment by integrating strategic human corrections while leveraging LLM's correctly labeled samples. Evaluations on HH-RLHF and TL;DR datasets show that RLTHF reaches full-human annotation-level alignment with only 6-7% of the human annotation effort. Furthermore, models trained on RLTHF's curated datasets for downstream tasks outperform those trained on fully human-annotated datasets, underscoring the effectiveness of RLTHF's strategic data curation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13416v1">Detecting LLM Fact-conflicting Hallucinations Enhanced by Temporal-logic-based Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 16 pages, under review. arXiv admin note: substantial text overlap with arXiv:2405.00648
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face the challenge of hallucinations -- outputs that seem coherent but are actually incorrect. A particularly damaging type is fact-conflicting hallucination (FCH), where generated content contradicts established facts. Addressing FCH presents three main challenges: 1) Automatically constructing and maintaining large-scale benchmark datasets is difficult and resource-intensive; 2) Generating complex and efficient test cases that the LLM has not been trained on -- especially those involving intricate temporal features -- is challenging, yet crucial for eliciting hallucinations; and 3) Validating the reasoning behind LLM outputs is inherently difficult, particularly with complex logical relationships, as it requires transparency in the model's decision-making process. This paper presents Drowzee, an innovative end-to-end metamorphic testing framework that utilizes temporal logic to identify fact-conflicting hallucinations (FCH) in large language models (LLMs). Drowzee builds a comprehensive factual knowledge base by crawling sources like Wikipedia and uses automated temporal-logic reasoning to convert this knowledge into a large, extensible set of test cases with ground truth answers. LLMs are tested using these cases through template-based prompts, which require them to generate both answers and reasoning steps. To validate the reasoning, we propose two semantic-aware oracles that compare the semantic structure of LLM outputs to the ground truths. Across nine LLMs in nine different knowledge domains, experimental results show that Drowzee effectively identifies rates of non-temporal-related hallucinations ranging from 24.7% to 59.8%, and rates of temporal-related hallucinations ranging from 16.7% to 39.2%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06647v4">How Efficient is LLM-Generated Code? A Rigorous & High-Standard Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has significantly pushed the frontiers of program synthesis. Advancement of LLM-based program synthesis calls for a thorough evaluation of LLM-generated code. Most evaluation frameworks focus on the (functional) correctness of generated code; efficiency, as an important measure of code quality, has been overlooked in existing evaluations. In this work, we develop ENAMEL (EfficeNcy AutoMatic EvaLuator), a rigorous and high-standard benchmark for evaluating the capability of LLMs in generating efficient code. Firstly, we propose a new efficiency metric called eff@k, which generalizes the pass@k metric from correctness to efficiency and appropriately handles right-censored execution time. Furthermore, we derive an unbiased and variance-reduced estimator of eff@k via Rao--Blackwellization; we also provide a numerically stable implementation for the new estimator. Secondly, to set a high-standard for efficiency evaluation, we employ a human expert to design best algorithms and implementations as our reference solutions of efficiency, many of which are much more efficient than existing canonical solutions in HumanEval and HumanEval+. Moreover, to ensure a rigorous evaluation, we employ a human expert to curate strong test case generators to filter out wrong code and differentiate suboptimal algorithms. An extensive study across 30 popular LLMs using our benchmark ENAMEL shows that LLMs still fall short of generating expert-level efficient code. Using two subsets of our problem set, we demonstrate that such deficiency is because current LLMs struggle in designing advanced algorithms and are barely aware of implementation optimization. Our benchmark is publicly available at https://github.com/q-rz/enamel .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18948v2">RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation through LLM Activation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) enriches the input to LLMs by retrieving information from the relevant knowledge database, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11433v3">FLAG-Trader: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) fine-tuned on multimodal financial data have demonstrated impressive reasoning capabilities in various financial tasks. However, they often struggle with multi-step, goal-oriented scenarios in interactive financial markets, such as trading, where complex agentic approaches are required to improve decision-making. To address this, we propose \textsc{FLAG-Trader}, a unified architecture integrating linguistic processing (via LLMs) with gradient-driven reinforcement learning (RL) policy optimization, in which a partially fine-tuned LLM acts as the policy network, leveraging pre-trained knowledge while adapting to the financial domain through parameter-efficient fine-tuning. Through policy gradient optimization driven by trading rewards, our framework not only enhances LLM performance in trading but also improves results on other financial-domain tasks. We present extensive empirical evidence to validate these enhancements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08631v2">Ensemble based approach to quantifying uncertainty of LLM based classifications</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The output of Large Language Models (LLMs) are a function of the internal model's parameters and the input provided into the context window. The hypothesis presented here is that under a greedy sampling strategy the variance in the LLM's output is a function of the conceptual certainty embedded in the model's parametric knowledge, as well as the lexical variance in the input. Finetuning the model results in reducing the sensitivity of the model output to the lexical input variations. This is then applied to a classification problem and a probabilistic method is proposed for estimating the certainties of the predicted classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13396v1">Prompting a Weighting Mechanism into LLM-as-a-Judge in Two-Step: A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 5 pages, 5 tables, 1 figure
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have emerged as promising tools for evaluating Natural Language Generation (NLG) tasks, their effectiveness is limited by their inability to appropriately weigh the importance of different topics, often overemphasizing minor details while undervaluing critical information, leading to misleading assessments. Our work proposes an efficient prompt design mechanism to address this specific limitation and provide a case study. Through strategic prompt engineering that incorporates explicit importance weighting mechanisms, we enhance using LLM-as-a-Judge ability to prioritize relevant information effectively, as demonstrated by an average improvement of 6% in the Human Alignment Rate (HAR) metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10844v2">Be Friendly, Not Friends: How LLM Sycophancy Shapes User Trust</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Recent studies have revealed that large language model (LLM)-powered conversational agents often exhibit `sycophancy', a tendency to adapt their responses to align with user perspectives, even at the expense of factual accuracy. However, users' perceptions of LLM sycophancy and its interplay with other anthropomorphic features (e.g., friendliness) in shaping user trust remains understudied. To bridge this gap, we conducted a 2 (Sycophancy: presence vs. absence) x 2 (Friendliness: high vs. low) between-subjects experiment (N = 224). Our study uncovered, for the first time, the intricate dynamics between LLM sycophancy and friendliness: When an LLM agent already exhibits a friendly demeanor, being sycophantic reduces perceived authenticity, thereby lowering user trust; Conversely, when the agent is less friendly, aligning its responses with user opinions makes it appear more genuine, leading to higher user trust. Our findings entail profound implications for AI persuasion through exploiting human psychological tendencies and highlight the imperative for responsible designs in user-LLM agent interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12945v2">LLMPopcorn: An Empirical Study of LLMs as Assistants for Popular Micro-video Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Popular Micro-videos, dominant on platforms like TikTok and YouTube, hold significant commercial value. The rise of high-quality AI-generated content has spurred interest in AI-driven micro-video creation. However, despite the advanced capabilities of large language models (LLMs) like ChatGPT and DeepSeek in text generation and reasoning, their potential to assist the creation of popular micro-videos remains largely unexplored. In this paper, we conduct an empirical study on LLM-assisted popular micro-video generation (LLMPopcorn). Specifically, we investigate the following research questions: (i) How can LLMs be effectively utilized to assist popular micro-video generation? (ii) To what extent can prompt-based enhancements optimize the LLM-generated content for higher popularity? (iii) How well do various LLMs and video generators perform in the popular micro-video generation task? By exploring these questions, we show that advanced LLMs like DeepSeek-V3 enable micro-video generation to achieve popularity comparable to human-created content. Prompt enhancements further boost popularity, and benchmarking highlights DeepSeek-V3 and DeepSeek-R1 among LLMs, while LTX-Video and HunyuanVideo lead in video generation. This pioneering work advances AI-assisted micro-video creation, uncovering new research opportunities. We will release the code and datasets to support future studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13358v1">Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, yet they still struggle with direct text editing tasks that demand precise, context-aware modifications. While models like ChatGPT excel in text generation and analysis, their editing abilities often fall short, addressing only superficial issues rather than deeper structural or logical inconsistencies. In this work, we introduce a dual approach to enhance LLMs editing performance. First, we present InstrEditBench, a high-quality benchmark dataset comprising over 20,000 structured editing tasks spanning Wiki articles, LaTeX documents, code, and database Domain-specific Languages (DSL). InstrEditBench is generated using an innovative automated workflow that accurately identifies and evaluates targeted edits, ensuring that modifications adhere strictly to specified instructions without altering unrelated content. Second, we propose FineEdit, a specialized model trained on this curated benchmark. Experimental results demonstrate that FineEdit achieves significant improvements around {10\%} compared with Gemini on direct editing tasks, convincingly validating its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10937v3">Proxona: Supporting Creators' Sensemaking and Ideation with LLM-Powered Audience Personas</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 Accepted by ACM CHI 2025; 32 pages (including 11 pages of Appendix); Acknowledgment added
    </div>
    <details class="paper-abstract">
      A content creator's success depends on understanding their audience, but existing tools fail to provide in-depth insights and actionable feedback necessary for effectively targeting their audience. We present Proxona, an LLM-powered system that transforms static audience comments into interactive, multi-dimensional personas, allowing creators to engage with them to gain insights, gather simulated feedback, and refine content. Proxona distills audience traits from comments, into dimensions (categories) and values (attributes), then clusters them into interactive personas representing audience segments. Technical evaluations show that Proxona generates diverse dimensions and values, enabling the creation of personas that sufficiently reflect the audience and support data grounded conversation. User evaluation with 11 creators confirmed that Proxona helped creators discover hidden audiences, gain persona-informed insights on early-stage content, and allowed them to confidently employ strategies when iteratively creating storylines. Proxona introduces a novel creator-audience interaction framework and fosters a persona-driven, co-creative process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13347v1">Craw4LLM: Efficient Web Crawling for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Web crawl is a main source of large language models' (LLMs) pretraining data, but the majority of crawled web pages are discarded in pretraining due to low data quality. This paper presents Crawl4LLM, an efficient web crawling method that explores the web graph based on the preference of LLM pretraining. Specifically, it leverages the influence of a webpage in LLM pretraining as the priority score of the web crawler's scheduler, replacing the standard graph connectivity based priority. Our experiments on a web graph containing 900 million webpages from a commercial search engine's index demonstrate the efficiency of Crawl4LLM in obtaining high-quality pretraining data. With just 21% URLs crawled, LLMs pretrained on Crawl4LLM data reach the same downstream performances of previous crawls, significantly reducing the crawling waste and alleviating the burdens on websites. Our code is publicly available at https://github.com/cxcscmu/Crawl4LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14145v1">LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 In submission to INTERSPEECH 2025
    </div>
    <details class="paper-abstract">
      Achieving full-duplex communication in spoken dialogue systems (SDS) requires real-time coordination between listening, speaking, and thinking. This paper proposes a semantic voice activity detection (VAD) module as a dialogue manager (DM) to efficiently manage turn-taking in full-duplex SDS. Implemented as a lightweight (0.5B) LLM fine-tuned on full-duplex conversation data, the semantic VAD predicts four control tokens to regulate turn-switching and turn-keeping, distinguishing between intentional and unintentional barge-ins while detecting query completion for handling user pauses and hesitations. By processing input speech in short intervals, the semantic VAD enables real-time decision-making, while the core dialogue engine (CDE) is only activated for response generation, reducing computational overhead. This design allows independent DM optimization without retraining the CDE, balancing interaction accuracy and inference efficiency for scalable, next-generation full-duplex SDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06556v3">ProjectTest: A Project-level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Unit test generation has become a promising and important use case of LLMs. However, existing evaluation benchmarks for assessing LLM unit test generation capabilities focus on function- or class-level code rather than more practical and challenging project-level codebases. To address such limitation, we propose ProjectTest, a project-level benchmark for unit test generation covering Python, Java, and JavaScript. ProjectTest features 20 moderate-sized and high-quality projects per language. We evaluate nine frontier LLMs on ProjectTest and the results show that all frontier LLMs tested exhibit moderate performance on ProjectTest on Python and Java, highlighting the difficulty of ProjectTest. We also conduct a thorough error analysis, which shows that even frontier LLMs, such as Claude-3.5-Sonnet, have significant basic yet critical errors, including compilation and cascade errors. Motivated by this observation, we further evaluate all frontier LLMs under manual error-fixing and self-error-fixing scenarios to assess their potential when equipped with error-fixing mechanisms. Our code and dataset is available at \href{https://github.com/YiboWANG214/ProjectTest}{ProjectTest}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10508v2">Prompt Engineering or Fine-Tuning: An Empirical Assessment of LLMs for Code</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 11 pages + reference. Accepted in 22nd International Conference on Mining Software Repositories, 2025. Technical Papers Track (MSR'25)
    </div>
    <details class="paper-abstract">
      The rapid advancements in large language models (LLMs) have greatly expanded the potential for automated code-related tasks. Two primary methodologies are used in this domain: prompt engineering and fine-tuning. Prompt engineering involves applying different strategies to query LLMs, like ChatGPT, while fine-tuning further adapts pre-trained models, such as CodeBERT, by training them on task-specific data. Despite the growth in the area, there remains a lack of comprehensive comparative analysis between the approaches for code models. In this paper, we evaluate GPT-4 using three prompt engineering strategies -- basic prompting, in-context learning, and task-specific prompting -- and compare it against 17 fine-tuned models across three code-related tasks: code summarization, generation, and translation. Our results indicate that GPT-4 with prompt engineering does not consistently outperform fine-tuned models. For instance, in code generation, GPT-4 is outperformed by fine-tuned models by 28.3% points on the MBPP dataset. It also shows mixed results for code translation tasks. Additionally, a user study was conducted involving 27 graduate students and 10 industry practitioners. The study revealed that GPT-4 with conversational prompts, incorporating human feedback during interaction, significantly improved performance compared to automated prompting. Participants often provided explicit instructions or added context during these interactions. These findings suggest that GPT-4 with conversational prompting holds significant promise for automated code-related tasks, whereas fully automated prompt engineering without human involvement still requires further investigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14133v1">Self-Regularization with Latent Space Explanations for Controllable LLM-based Classification</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 Pre-print, 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Modern text classification methods heavily rely on contextual embeddings from large language models (LLMs). Compared to human-engineered features, these embeddings provide automatic and effective representations for classification model training. However, they also introduce a challenge: we lose the ability to manually remove unintended features, such as sensitive or task-irrelevant features, to guarantee regulatory compliance or improve the generalizability of classification models. This limitation arises because LLM embeddings are opaque and difficult to interpret. In this paper, we propose a novel framework to identify and regularize unintended features in the LLM latent space. Specifically, we first pre-train a sparse autoencoder (SAE) to extract interpretable features from LLM latent spaces. To ensure the SAE can capture task-specific features, we further fine-tune it on task-specific datasets. In training the classification model, we propose a simple and effective regularizer, by minimizing the similarity between the classifier weights and the identified unintended feature, to remove the impacts of these unintended features toward classification. We evaluate the proposed framework on three real-world tasks, including toxic chat detection, reward modeling, and disease diagnosis. Results show that the proposed framework can significantly improve the classifier's generalizability by regularizing those features that are not semantically correlated to each task. This work pioneers controllable text classification on LLM latent spaces by leveraging interpreted features to address generalizability, fairness, and privacy challenges. We will release our code and data once accepted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12003v2">RAG-Optimized Tibetan Tourism LLMs: Enhancing Accuracy and Personalization</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 Accepted by AIPR 2024
    </div>
    <details class="paper-abstract">
      With the development of the modern social economy, tourism has become an important way to meet people's spiritual needs, bringing development opportunities to the tourism industry. However, existing large language models (LLMs) face challenges in personalized recommendation capabilities and the generation of content that can sometimes produce hallucinations. This study proposes an optimization scheme for Tibet tourism LLMs based on retrieval-augmented generation (RAG) technology. By constructing a database of tourist viewpoints and processing the data using vectorization techniques, we have significantly improved retrieval accuracy. The application of RAG technology effectively addresses the hallucination problem in content generation. The optimized model shows significant improvements in fluency, accuracy, and relevance of content generation. This research demonstrates the potential of RAG technology in the standardization of cultural tourism information and data analysis, providing theoretical and technical support for the development of intelligent cultural tourism service systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14127v1">Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 In-progress preprint
    </div>
    <details class="paper-abstract">
      Multiple choice question answering (MCQA) is popular for LLM evaluation due to its simplicity and human-like testing, but we argue for its reform. We first reveal flaws in MCQA's format, as it struggles to: 1) test generation/subjectivity; 2) match LLM use cases; and 3) fully test knowledge. We instead advocate for generative formats based on human testing-where LLMs construct and explain answers-better capturing user needs and knowledge while remaining easy to score. We then show even when MCQA is a useful format, its datasets suffer from: leakage; unanswerability; shortcuts; and saturation. In each issue, we give fixes from education, like rubrics to guide MCQ writing; scoring methods to bridle guessing; and Item Response Theory to build harder MCQs. Lastly, we discuss LLM errors in MCQA-robustness, biases, and unfaithful explanations-showing how our prior solutions better measure or address these issues. While we do not need to desert MCQA, we encourage more efforts in refining the task based on educational testing, advancing evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14924v1">A Tale of Two Structures: Do LLMs Capture the Fractal Complexity of Language?</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Language exhibits a fractal structure in its information-theoretic complexity (i.e. bits per token), with self-similarity across scales and long-range dependence (LRD). In this work, we investigate whether large language models (LLMs) can replicate such fractal characteristics and identify conditions-such as temperature setting and prompting method-under which they may fail. Moreover, we find that the fractal parameters observed in natural language are contained within a narrow range, whereas those of LLMs' output vary widely, suggesting that fractal parameters might prove helpful in detecting a non-trivial portion of LLM-generated texts. Notably, these findings, and many others reported in this work, are robust to the choice of the architecture; e.g. Gemini 1.0 Pro, Mistral-7B and Gemma-2B. We also release a dataset comprising of over 240,000 articles generated by various LLMs (both pretrained and instruction-tuned) with different decoding temperatures and prompting methods, along with their corresponding human-generated texts. We hope that this work highlights the complex interplay between fractal properties, prompting, and statistical mimicry in LLMs, offering insights for generating, evaluating and detecting synthetic texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14922v1">SIFT: Grounding LLM Reasoning in Contexts via Stickers</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      This paper identifies the misinterpretation of the context can be a significant issue during the reasoning process of large language models, spanning from smaller models like Llama3.2-3B-Instruct to cutting-edge ones like DeepSeek-R1. For example, in the phrase "10 dollars per kilo," LLMs might not recognize that "per" means "for each," leading to calculation errors. We introduce a novel, post-training approach called **Stick to the Facts (SIFT)** to tackle this. SIFT leverages increasing inference-time compute to ground LLM reasoning in contexts. At the core of SIFT lies the *Sticker*, which is generated by the model itself to explicitly emphasize the key information within the context. Given the curated Sticker, SIFT generates two predictions -- one from the original query and one from the query augmented with the Sticker. If they differ, the Sticker is sequentially refined via *forward* optimization (to better align the extracted facts with the query) and *inverse* generation (to conform with the model's inherent tendencies) for more faithful reasoning outcomes. Studies across diverse models (from 3B to 100B+) and benchmarks (e.g., GSM8K, MATH-500) reveal consistent performance improvements. Notably, SIFT improves the pass@1 accuracy of DeepSeek-R1 on AIME2024 from 78.33% to **85.67**%, establishing a new state-of-the-art in the open-source community. The code is available at https://github.com/zhijie-group/SIFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14921v1">The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      How much information about training samples can be gleaned from synthetic data generated by Large Language Models (LLMs)? Overlooking the subtleties of information flow in synthetic data generation pipelines can lead to a false sense of privacy. In this paper, we design membership inference attacks (MIAs) that target data used to fine-tune pre-trained LLMs that are then used to synthesize data, particularly when the adversary does not have access to the fine-tuned model but only to the synthetic data. We show that such data-based MIAs do significantly better than a random guess, meaning that synthetic data leaks information about the training data. Further, we find that canaries crafted to maximize vulnerability to model-based MIAs are sub-optimal for privacy auditing when only synthetic data is released. Such out-of-distribution canaries have limited influence on the model's output when prompted to generate useful, in-distribution synthetic data, which drastically reduces their vulnerability. To tackle this problem, we leverage the mechanics of auto-regressive models to design canaries with an in-distribution prefix and a high-perplexity suffix that leave detectable traces in synthetic data. This enhances the power of data-based MIAs and provides a better assessment of the privacy risks of releasing synthetic data generated by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14910v1">EvoP: Robust LLM Inference via Evolutionary Pruning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in natural language processing tasks, but their massive size and computational demands hinder their deployment in resource-constrained environments. Existing structured pruning methods address this issue by removing redundant structures (e.g., elements, channels, layers) from the model. However, these methods employ a heuristic pruning strategy, which leads to suboptimal performance. Besides, they also ignore the data characteristics when pruning the model. To overcome these limitations, we propose EvoP, an evolutionary pruning framework for robust LLM inference. EvoP first presents a cluster-based calibration dataset sampling (CCDS) strategy for creating a more diverse calibration dataset. EvoP then introduces an evolutionary pruning pattern searching (EPPS) method to find the optimal pruning pattern. Compared to existing structured pruning techniques, EvoP achieves the best performance while maintaining the best efficiency. Experiments across different LLMs and different downstream tasks validate the effectiveness of the proposed EvoP, making it a practical and scalable solution for deploying LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14907v1">GneissWeb: Preparing High Quality Data for LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Data quantity and quality play a vital role in determining the performance of Large Language Models (LLMs). High-quality data, in particular, can significantly boost the LLM's ability to generalize on a wide range of downstream tasks. Large pre-training datasets for leading LLMs remain inaccessible to the public, whereas many open datasets are small in size (less than 5 trillion tokens), limiting their suitability for training large models. In this paper, we introduce GneissWeb, a large dataset yielding around 10 trillion tokens that caters to the data quality and quantity requirements of training LLMs. Our GneissWeb recipe that produced the dataset consists of sharded exact sub-string deduplication and a judiciously constructed ensemble of quality filters. GneissWeb achieves a favorable trade-off between data quality and quantity, producing models that outperform models trained on state-of-the-art open large datasets (5+ trillion tokens). We show that models trained using GneissWeb dataset outperform those trained on FineWeb-V1.1.0 by 2.73 percentage points in terms of average score computed on a set of 11 commonly used benchmarks (both zero-shot and few-shot) for pre-training dataset evaluation. When the evaluation set is extended to 20 benchmarks (both zero-shot and few-shot), models trained using GneissWeb still achieve a 1.75 percentage points advantage over those trained on FineWeb-V1.1.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14043v3">Taxonomy-Guided Zero-Shot Recommendations with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      With the emergence of large language models (LLMs) and their ability to perform a variety of tasks, their application in recommender systems (RecSys) has shown promise. However, we are facing significant challenges when deploying LLMs into RecSys, such as limited prompt length, unstructured item information, and un-constrained generation of recommendations, leading to sub-optimal performance. To address these issues, we propose a novel method using a taxonomy dictionary. This method provides a systematic framework for categorizing and organizing items, improving the clarity and structure of item information. By incorporating the taxonomy dictionary into LLM prompts, we achieve efficient token utilization and controlled feature generation, leading to more accurate and contextually relevant recommendations. Our Taxonomy-guided Recommendation (TaxRec) approach features a two-step process: one-time taxonomy categorization and LLM-based recommendation, enabling zero-shot recommendations without the need for domain-specific fine-tuning. Experimental results demonstrate TaxRec significantly enhances recommendation quality compared to traditional zero-shot approaches, showcasing its efficacy as personal recommender with LLMs. Code is available at https://github.com/yueqingliang1/TaxRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14122v1">Benchmarking LLMs for Political Science: A United Nations Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant advances in natural language processing, yet their potential for high-stake political decision-making remains largely unexplored. This paper addresses the gap by focusing on the application of LLMs to the United Nations (UN) decision-making process, where the stakes are particularly high and political decisions can have far-reaching consequences. We introduce a novel dataset comprising publicly available UN Security Council (UNSC) records from 1994 to 2024, including draft resolutions, voting records, and diplomatic speeches. Using this dataset, we propose the United Nations Benchmark (UNBench), the first comprehensive benchmark designed to evaluate LLMs across four interconnected political science tasks: co-penholder judgment, representative voting simulation, draft adoption prediction, and representative statement generation. These tasks span the three stages of the UN decision-making process--drafting, voting, and discussing--and aim to assess LLMs' ability to understand and simulate political dynamics. Our experimental analysis demonstrates the potential and challenges of applying LLMs in this domain, providing insights into their strengths and limitations in political science. This work contributes to the growing intersection of AI and political science, opening new avenues for research and practical applications in global governance. The UNBench Repository can be accessed at: https://github.com/yueqingliang1/UNBench.
    </details>
</div>
