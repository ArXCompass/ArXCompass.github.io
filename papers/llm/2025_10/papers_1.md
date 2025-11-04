# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27675v1">On Selecting Few-Shot Examples for LLM-based Code Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities for many coding tasks, including summarization, translation, completion, and code generation. However, detecting code vulnerabilities remains a challenging task for LLMs. An effective way to improve LLM performance is in-context learning (ICL) - providing few-shot examples similar to the query, along with correct answers, can improve an LLM's ability to generate correct solutions. However, choosing the few-shot examples appropriately is crucial to improving model performance. In this paper, we explore two criteria for choosing few-shot examples for ICL used in the code vulnerability detection task. The first criterion considers if the LLM (consistently) makes a mistake or not on a sample with the intuition that LLM performance on a sample is informative about its usefulness as a few-shot example. The other criterion considers similarity of the examples with the program under query and chooses few-shot examples based on the $k$-nearest neighbors to the given sample. We perform evaluations to determine the benefits of these criteria individually as well as under various combinations, using open-source models on multiple datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27656v1">RDMA Point-to-Point Communication for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Emerging Large Language Model (LLM) system patterns, such as disaggregated inference, Mixture-of-Experts (MoE) routing, and asynchronous reinforcement fine-tuning, require flexible point-to-point communication beyond simple collectives. Existing implementations are locked to specific Network Interface Controllers (NICs), hindering integration into inference engines and portability across hardware providers. We present TransferEngine, which bridges the functionality of common NICs to expose a uniform interface. TransferEngine exposes one-sided WriteImm operations with a ImmCounter primitive for completion notification, without ordering assumptions of network transport, transparently managing multiple NICs per GPU. We demonstrate peak throughput of 400 Gbps on both NVIDIA ConnectX-7 and AWS Elastic Fabric Adapter (EFA). We showcase TransferEngine through three production systems: (1) KvCache transfer for disaggregated inference with dynamic scaling, (2) RL weight updates achieving 1.3 seconds for trillion-parameter models, and (3) MoE dispatch/combine implementation exceeding DeepEP decode latency on ConnectX-7, with the first viable latencies on EFA. We demonstrate that our portable point-to-point communication complements collectives while avoiding lock-in.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27610v1">ORGEval: Graph-Theoretic Evaluation of LLMs in Optimization Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Formulating optimization problems for industrial applications demands significant manual effort and domain expertise. While Large Language Models (LLMs) show promise in automating this process, evaluating their performance remains difficult due to the absence of robust metrics. Existing solver-based approaches often face inconsistency, infeasibility issues, and high computational costs. To address these issues, we propose ORGEval, a graph-theoretic evaluation framework for assessing LLMs' capabilities in formulating linear and mixed-integer linear programs. ORGEval represents optimization models as graphs, reducing equivalence detection to graph isomorphism testing. We identify and prove a sufficient condition, when the tested graphs are symmetric decomposable (SD), under which the Weisfeiler-Lehman (WL) test is guaranteed to correctly detect isomorphism. Building on this, ORGEval integrates a tailored variant of the WL-test with an SD detection algorithm to evaluate model equivalence. By focusing on structural equivalence rather than instance-level configurations, ORGEval is robust to numerical variations. Experimental results show that our method can successfully detect model equivalence and produce 100\% consistent results across random parameter configurations, while significantly outperforming solver-based methods in runtime, especially on difficult problems. Leveraging ORGEval, we construct the Bench4Opt dataset and benchmark state-of-the-art LLMs on optimization modeling. Our results reveal that although optimization modeling remains challenging for all LLMs, DeepSeek-V3 and Claude-Opus-4 achieve the highest accuracies under direct prompting, outperforming even leading reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27598v1">InnovatorBench: Evaluating Agents' Ability to Conduct Innovative LLM Research</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      AI agents could accelerate scientific discovery by automating hypothesis formation, experiment design, coding, execution, and analysis, yet existing benchmarks probe narrow skills in simplified settings. To address this gap, we introduce InnovatorBench, a benchmark-platform pair for realistic, end-to-end assessment of agents performing Large Language Model (LLM) research. It comprises 20 tasks spanning Data Construction, Filtering, Augmentation, Loss Design, Reward Design, and Scaffold Construction, which require runnable artifacts and assessment of correctness, performance, output quality, and uncertainty. To support agent operation, we develop ResearchGym, a research environment offering rich action spaces, distributed and long-horizon execution, asynchronous monitoring, and snapshot saving. We also implement a lightweight ReAct agent that couples explicit reasoning with executable planning using frontier models such as Claude-4, GPT-5, GLM-4.5, and Kimi-K2. Our experiments demonstrate that while frontier models show promise in code-driven research tasks, they struggle with fragile algorithm-related tasks and long-horizon decision making, such as impatience, poor resource management, and overreliance on template-based reasoning. Furthermore, agents require over 11 hours to achieve their best performance on InnovatorBench, underscoring the benchmark's difficulty and showing the potential of InnovatorBench to be the next generation of code-based research benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10724v2">HELIOS: Adaptive Model And Early-Exit Selection for Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Early-Exit Large Language Models (EE-LLMs) enable high throughput inference by allowing tokens to exit early at intermediate layers. However, their throughput is limited by the computational and memory savings. Existing EE-LLM frameworks rely on a single model and therefore, their token generation latencies are bottlenecked by tokens that do not exit early and traverse additional layers. Moreover, early exits are only known at runtime and depend on the request. Therefore, these frameworks load the weights of all model layers even though large portions remain unused when tokens exit early. The lack of memory savings limit us from scaling the batch sizes. We propose $\textit{HELIOS}$, a framework that improves both token generation latency and batch sizes to enable high-throughput in EE-LLMs. HELIOS exploits two insights. $\textit{First}$, early exits are often complimentary across models, tokens that do not exit early on one model often take an early-exit on another. HELIOS employs multiple models and dynamically switches between them to collectively maximize the number of tokens that exit early, and minimize token generation latencies. $\textit{Second}$, even when a predicted token does not exit early due to poor confidence, it often remains unchanged even after additional layer traversal. HELIOS greedily allows such tokens to exit early and only loads the weights of the most likely to be used layers, yielding memory savings which is then re-purposed to increase batch sizes. HELIOS employs real-time profiling to accurately identify the early-exit distributions, and adaptively switches between models by tracking tokens in real-time to minimize the performance degradation caused by greedy model loading and exiting. Our evaluations show that HELIOS achieves $1.48\times$ higher throughput and $15.14\times$ larger batch size compared to existing EE-LLM frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27556v1">Data-Efficient Domain Adaptation for LLM-based MT using Contrastive Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      LLMs often require adaptation to domain-specific requirements, a process that can be expensive when relying solely on SFT. We present an empirical study on applying CPO to simulate a post-editing workflow for data-efficient domain adaptation. Our approach synthesizes preference pairs by treating the base model's own raw output as the 'rejected' translation and the human-approved TM entry as the 'chosen' one. This method provides direct feedback on the model's current knowledge, guiding it to align with domain-specific standards. Experiments in English-Brazilian Portuguese and English-Korean show that, by using just 14.7k preference pairs, the model achieves performance close to that of a model trained on 160k+ samples with SFT, demonstrating significant data efficiency. Although we showcase its effectiveness in MT, this application of CPO naturally generalizes to other generative tasks where a model's initial drafts can serve as a contrastive signal against a golden reference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27521v1">Independent Clinical Evaluation of General-Purpose LLM Responses to Signals of Suicide Risk</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      We introduce findings and methods to facilitate evidence-based discussion about how large language models (LLMs) should behave in response to user signals of risk of suicidal thoughts and behaviors (STB). People are already using LLMs as mental health resources, and several recent incidents implicate LLMs in mental health crises. Despite growing attention, few studies have been able to effectively generalize clinical guidelines to LLM use cases, and fewer still have proposed methodologies that can be iteratively applied as knowledge improves about the elements of human-AI interaction most in need of study. We introduce an assessment of LLM alignment with guidelines for ethical communication, adapted from clinical principles and applied to expressions of risk factors for STB in multi-turn conversations. Using a codebook created and validated by clinicians, mobilizing the volunteer participation of practicing therapists and trainees (N=43) based in the U.S., and using generalized linear mixed-effects models for statistical analysis, we assess a single fully open-source LLM, OLMo-2-32b. We show how to assess when a model deviates from clinically informed guidelines in a way that may pose a hazard and (thanks to its open nature) facilitates future investigation as to why. We find that contrary to clinical best practice, OLMo-2-32b, and, possibly by extension, other LLMs, will become less likely to invite continued dialog as users send more signals of STB risk in multi-turn settings. We also show that OLMo-2-32b responds differently depending on the risk factor expressed. This empirical evidence highlights that just as chatbots pose hazards if their responses reinforce delusions or assist in suicidal acts, they may also discourage further help-seeking or cause feelings of dismissal or abandonment by withdrawing from conversations when STB risk is expressed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27489v1">Auditing LLM Editorial Bias in News Media Exposure</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 Under Peer Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly act as gateways to web content, shaping how millions of users encounter online information. Unlike traditional search engines, whose retrieval and ranking mechanisms are well studied, the selection processes of web-connected LLMs add layers of opacity to how answers are generated. By determining which news outlets users see, these systems can influence public opinion, reinforce echo chambers, and pose risks to civic discourse and public trust. This work extends two decades of research in algorithmic auditing to examine how LLMs function as news engines. We present the first audit comparing three leading agents, GPT-4o-Mini, Claude-3.7-Sonnet, and Gemini-2.0-Flash, against Google News, asking: \textit{How do LLMs differ from traditional aggregators in the diversity, ideology, and reliability of the media they expose to users?} Across 24 global topics, we find that, compared to Google News, LLMs surface significantly fewer unique outlets and allocate attention more unevenly. In the same way, GPT-4o-Mini emphasizes more factual and right-leaning sources; Claude-3.7-Sonnet favors institutional and civil-society domains and slightly amplifies right-leaning exposure; and Gemini-2.0-Flash exhibits a modest left-leaning tilt without significant changes in factuality. These patterns remain robust under prompt variations and alternative reliability benchmarks. Together, our findings show that LLMs already enact \textit{agentic editorial policies}, curating information in ways that diverge from conventional aggregators. Understanding and governing their emerging editorial power will be critical for ensuring transparency, pluralism, and trust in digital information ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05831v2">Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04412v2">Beyond Pixels: Exploring DOM Downsampling for LLM-Based Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 20 pages, LaTeX; repository URL updated, typos corrected
    </div>
    <details class="paper-abstract">
      Frontier LLMs only recently enabled serviceable, autonomous web agents. At that, a model poses as an instantaneous domain model backend. Ought to suggest interaction, it is consulted with a web-based task and respective application state. The key problem lies in application state serialisation - referred to as snapshot. State-of-the-art web agents are premised on grounded GUI snapshots, i.e., screenshots enhanced with visual cues. Not least to resemble human perception, but for images representing relatively cheap means of model input. LLM vision still lag behind code interpretation capabilities. DOM snapshots, which structurally resemble HTML, impose a desired alternative. Vast model input token size, however, disables reliable implementation with web agents to date. We propose D2Snap, a first-of-its-kind DOM downsampling algorithm. Based on a GPT-4o backend, we evaluate D2Snap on tasks sampled from the Online-Mind2Web dataset. The success rate of D2Snap-downsampled DOM snapshots (67%) matches a grounded GUI snapshot baseline (65%) - within the same input token order of magnitude (1e3). Our best evaluated configurations - one token order above, but within the model's context window - outperform this baseline by 8%. Our evaluation, moreover, yields that DOM-inherent hierarchy embodies a strong UI feature for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14216v2">Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Recent studies have shown that reinforcement learning with verifiable rewards (RLVR) enhances overall accuracy (pass@1) but often fails to improve capability (pass@k) of LLMs in reasoning tasks, while distillation can improve both. In this paper, we investigate the mechanisms behind these phenomena. First, we demonstrate that RLVR struggles to improve capability as it focuses on improving the accuracy of the easier questions to the detriment of the accuracy of the most difficult questions. Second, we show that RLVR does not merely increase the success probability for the easier questions, but in our small model settings, produces quality responses that were absent in its original output distribution. In addition, we show these responses are neither noticeably longer nor feature more reflection-related keywords, underscoring the need for more reliable indicators of response quality. Third, from the experiment distilling teacher responses to in-distribution problems, we find that capability does not always improve with distillation. We conjecture that capability improves only when new knowledge is introduced, whereas distilling reasoning patterns only improves accuracy but not capability, sacrificing performance on the most difficult questions, similar to RLVR. Together, these findings offer a clearer understanding of how RLVR and distillation shape reasoning behavior in LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27417v1">Agentic LLMs for REST API Test Amplification: A Comparative Study Across Cloud Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Representational State Transfer (REST) APIs are a cornerstone of modern cloud native systems. Ensuring their reliability demands automated test suites that exercise diverse and boundary level behaviors. Nevertheless, designing such test cases remains a challenging and resource intensive endeavor. This study extends prior work on Large Language Model (LLM) based test amplification by evaluating single agent and multi agent configurations across four additional cloud applications. The amplified test suites maintain semantic validity with minimal human intervention. The results demonstrate that agentic LLM systems can effectively generalize across heterogeneous API architectures, increasing endpoint and parameter coverage while revealing defects. Moreover, a detailed analysis of computational cost, runtime, and energy consumption highlights trade-offs between accuracy, scalability, and efficiency. These findings underscore the potential of LLM driven test amplification to advance the automation and sustainability of REST API testing in complex cloud environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27400v1">Balancing Knowledge Updates: Toward Unified Modular Editing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Knowledge editing has emerged as an efficient approach for updating factual knowledge in large language models (LLMs). It typically locates knowledge storage modules and then modifies their parameters. However, most existing methods focus on the weights of multilayer perceptron (MLP) modules, which are often identified as the main repositories of factual information. Other components, such as attention (Attn) modules, are often ignored during editing. This imbalance can leave residual outdated knowledge and limit editing effectiveness. We perform comprehensive knowledge localization experiments on advanced LLMs and find that Attn modules play a substantial role in factual knowledge storage and retrieval, especially in earlier layers. Based on these insights, we propose IntAttn-Edit, a method that extends the associative memory paradigm to jointly update both MLP and Attn modules. Our approach uses a knowledge balancing strategy that allocates update magnitudes in proportion to each module's measured contribution to knowledge storage. Experiments on standard benchmarks show that IntAttn-Edit achieves higher edit success, better generalization, and stronger knowledge preservation than prior methods. Further analysis shows that the balancing strategy keeps editing performance within an optimal range across diverse settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07877v2">Ready to Translate, Not to Represent? Bias and Performance Gaps in Multilingual LLMs Across Language Families and Domains</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has redefined Machine Translation (MT), enabling context-aware and fluent translations across hundreds of languages and textual domains. Despite their remarkable capabilities, LLMs often exhibit uneven performance across language families and specialized domains. Moreover, recent evidence reveals that these models can encode and amplify different biases present in their training data, posing serious concerns for fairness, especially in low-resource languages. To address these gaps, we introduce Translation Tangles, a unified framework and dataset for evaluating the translation quality and fairness of open-source LLMs. Our approach benchmarks 24 bidirectional language pairs across multiple domains using different metrics. We further propose a hybrid bias detection pipeline that integrates rule-based heuristics, semantic similarity filtering, and LLM-based validation. We also introduce a high-quality, bias-annotated dataset based on human evaluations of 1,439 translation-reference pairs. The code and dataset are accessible on GitHub: https://github.com/faiyazabdullah/TranslationTangles
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27355v1">ThoughtProbe: Classifier-Guided LLM Thought Space Exploration via Probing Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 EMNLP2025 main conference
    </div>
    <details class="paper-abstract">
      This paper introduces ThoughtProbe, a novel inference time framework that leverages the hidden reasoning features of Large Language Models (LLMs) to improve their reasoning performance. Unlike previous works that manipulate the hidden representations to steer LLM generation, we harness them as discriminative signals to guide the tree structured response space exploration. In each node expansion, a classifier serves as a scoring and ranking mechanism that efficiently allocates computational resources by prioritizing higher score candidates for continuation. After completing the tree expansion, we collect answers from all branches to form a candidate answer pool. We then propose a branch aggregation method that marginalizes over all supporting branches by aggregating their CoT scores, thereby identifying the optimal answer from the pool. Experimental results show that our framework's comprehensive exploration not only covers valid reasoning chains but also effectively identifies them, achieving significant improvements across multiple arithmetic reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27353v1">An In-depth Study of LLM Contributions to the Bin Packing Problem</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 15 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Recent studies have suggested that Large Language Models (LLMs) could provide interesting ideas contributing to mathematical discovery. This claim was motivated by reports that LLM-based genetic algorithms produced heuristics offering new insights into the online bin packing problem under uniform and Weibull distributions. In this work, we reassess this claim through a detailed analysis of the heuristics produced by LLMs, examining both their behavior and interpretability. Despite being human-readable, these heuristics remain largely opaque even to domain experts. Building on this analysis, we propose a new class of algorithms tailored to these specific bin packing instances. The derived algorithms are significantly simpler, more efficient, more interpretable, and more generalizable, suggesting that the considered instances are themselves relatively simple. We then discuss the limitations of the claim regarding LLMs' contribution to this problem, which appears to rest on the mistaken assumption that the instances had previously been studied. Our findings instead emphasize the need for rigorous validation and contextualization when assessing the scientific value of LLM-generated outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03419v2">Curse of Knowledge: When Complex Evaluation Context Benefits yet Biases LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow more capable, they face increasingly diverse and complex tasks, making reliable evaluation challenging. The paradigm of LLMs as judges has emerged as a scalable solution, yet prior work primarily focuses on simple settings. Their reliability in complex tasks--where multi-faceted rubrics, unstructured reference answers, and nuanced criteria are critical--remains understudied. In this paper, we constructed ComplexEval, a challenge benchmark designed to systematically expose and quantify Auxiliary Information Induced Biases. We systematically investigated and validated 6 previously unexplored biases across 12 basic and 3 advanced scenarios. Key findings reveal: (1) all evaluated models exhibit significant susceptibility to these biases, with bias magnitude scaling with task complexity; (2) notably, Large Reasoning Models (LRMs) show paradoxical vulnerability. Our in-depth analysis offers crucial insights for improving the accuracy and verifiability of evaluation signals, paving the way for more general and robust evaluation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00920v2">Can Emulating Semantic Translation Help LLMs with Code Translation? A Study Based on Pseudocode</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show great potential in code translation. However, accurate translation remains challenging when using the commonly adopted direct code-to-code translation approach, which converts a program into the target programming language (PL) in a single step. Inspired by the success of incorporating intermediate steps to guide LLMs in resolving challenging tasks, we explore pseudocode-based code translation, which emulates the human semantic translation by first interpreting the program's intent and logic into pseudocode and then implementing it in the target PL. We find that pseudocode-based translation helps translate programs that direct translation struggles to handle. Nonetheless, the effectiveness, advantages, and limitations of this approach remain underexplored. To bridge this gap, we present an empirical study on pseudocode-based code translation, aiming to investigate its effectiveness in enhancing the direct translation approach, illuminate its effective usage, and identify limitations hindering its potential benefits. By comparing direct and pseudocode-based translation approaches on 9,690 translation tasks across six PLs with five popular LLMs, we demonstrate that pseudocode-based translation can effectively complement direct translation, particularly when translating from flexible to rigid PLs or dealing with low-resource Rust. Based on these findings, we suggest adopting strategies that combine the complementary strengths of both approaches to enhance code translation accuracy. We also reveal the advantages of pseudocode-based translation in disentangling translations of complicated programs and mitigating distractions from detailed implementations in original programs, as well as its limitations due to incorrect, incomplete, or ambiguous pseudocode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27287v1">Can LLMs Help You at Work? A Sandbox for Evaluating LLM Agents in Enterprise Environments</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 Accepted at EMNLP 2025 Main Track
    </div>
    <details class="paper-abstract">
      Enterprise systems are crucial for enhancing productivity and decision-making among employees and customers. Integrating LLM based systems into enterprise systems enables intelligent automation, personalized experiences, and efficient information retrieval, driving operational efficiency and strategic growth. However, developing and evaluating such systems is challenging due to the inherent complexity of enterprise environments, where data is fragmented across multiple sources and governed by sophisticated access controls. We present EnterpriseBench, a comprehensive benchmark that simulates enterprise settings, featuring 500 diverse tasks across software engineering, HR, finance, and administrative domains. Our benchmark uniquely captures key enterprise characteristics including data source fragmentation, access control hierarchies, and cross-functional workflows. Additionally, we provide a novel data generation pipeline that creates internally consistent enterprise tasks from organizational metadata. Experiments with state-of-the-art LLM agents demonstrate that even the most capable models achieve only 41.8% task completion, highlighting significant opportunities for improvement in enterprise-focused AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09045v2">LLM Based Long Code Translation using Identifier Replacement</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      In the domain of software development, LLMs have been utilized to automate tasks such as code translation, where source code from one programming language is translated to another while preserving its functionality. However, LLMs often struggle with long source codes that don't fit into the context window, which produces inaccurate translations. To address this, we propose a novel zero-shot code translation method that incorporates identifier replacement. By substituting user-given long identifiers with generalized placeholders during translation, our method allows the LLM to focus on the logical structure of the code, by reducing token count and memory usage, which improves the efficiency and cost-effectiveness of long code translation. Our empirical results demonstrate that our approach preserves syntactical and hierarchical information and produces translation results with reduced tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13178v5">SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 28 pages, 19 tables, 15 figures
    </div>
    <details class="paper-abstract">
      With the integration of large language models (LLMs), embodied agents have strong capabilities to understand and plan complicated natural language instructions. However, a foreseeable issue is that those embodied agents can also flawlessly execute some hazardous tasks, potentially causing damages in the real world. Existing benchmarks predominantly overlook critical safety risks, focusing solely on planning performance, while a few evaluate LLMs' safety awareness only on non-interactive image-text data. To address this gap, we present SafeAgentBench -- the first comprehensive benchmark for safety-aware task planning of embodied LLM agents in interactive simulation environments, covering both explicit and implicit hazards. SafeAgentBench includes: (1) an executable, diverse, and high-quality dataset of 750 tasks, rigorously curated to cover 10 potential hazards and 3 task types; (2) SafeAgentEnv, a universal embodied environment with a low-level controller, supporting multi-agent execution with 17 high-level actions for 9 state-of-the-art baselines; and (3) reliable evaluation methods from both execution and semantic perspectives. Experimental results show that, although agents based on different design frameworks exhibit substantial differences in task success rates, their overall safety awareness remains weak. The most safety-conscious baseline achieves only a 10% rejection rate for detailed hazardous tasks. Moreover, simply replacing the LLM driving the agent does not lead to notable improvements in safety awareness. Dataset and codes are available in https://github.com/shengyin1224/SafeAgentBench and https://huggingface.co/datasets/safeagentbench/SafeAgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05079v3">LLM-Guided Scenario-based GUI Testing</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      The assurance of mobile app GUIs has become increasingly important, as the GUI serves as the primary medium of interaction between users and apps. Although numerous automated GUI testing approaches have been developed with diverse strategies, a substantial gap remains between these approaches and the underlying app business logic. Most existing approaches focus on general exploration rather than the completion of specific testing scenarios, often missing critical functionalities. Inspired by manual testing, which treats business logic-driven scenarios as the fundamental unit of testing, this paper introduces an approach that leverages large language models to comprehend GUI semantics and contextual relevance to given scenarios. Building on this capability, we propose ScenGen, an LLM-guided scenario-based GUI testing framework employing multi-agent collaboration to simulate and automate manual testing phases. Specifically, ScenGen integrates five agents: the Observer, Decider, Executor, Supervisor, and Recorder. The Observer perceives the app GUI state by extracting and structuring GUI widgets and layouts, interpreting semantic information. This is passed to the Decider, which makes scenario-driven decisions with LLM guidance to identify target widgets and determine actions toward fulfilling specific goals. The Executor performs these operations, while the Supervisor verifies alignment with intended scenario completion, ensuring traceability and consistency. Finally, the Recorder logs GUI operations into context memory as a knowledge base for subsequent decision-making and monitors runtime bugs. Comprehensive evaluations demonstrate that ScenGen effectively generates scenario-based GUI tests guided by LLM collaboration, achieving higher relevance to business logic and improving the completeness of automated GUI testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27246v1">Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Evaluating the abilities of large language models (LLMs) for tasks that require long-term memory and thus long-context reasoning, for example in conversational settings, is hampered by the existing benchmarks, which often lack narrative coherence, cover narrow domains, and only test simple recall-oriented tasks. This paper introduces a comprehensive solution to these challenges. First, we present a novel framework for automatically generating long (up to 10M tokens), coherent, and topically diverse conversations, accompanied by probing questions targeting a wide range of memory abilities. From this, we construct BEAM, a new benchmark comprising 100 conversations and 2,000 validated questions. Second, to enhance model performance, we propose LIGHT-a framework inspired by human cognition that equips LLMs with three complementary memory systems: a long-term episodic memory, a short-term working memory, and a scratchpad for accumulating salient facts. Our experiments on BEAM reveal that even LLMs with 1M token context windows (with and without retrieval-augmentation) struggle as dialogues lengthen. In contrast, LIGHT consistently improves performance across various models, achieving an average improvement of 3.5%-12.69% over the strongest baselines, depending on the backbone LLM. An ablation study further confirms the contribution of each memory component.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02901v3">A Comprehensive Survey on Process-Oriented Automatic Text Summarization with Exploration of LLM-Based Methods</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Automatic Text Summarization (ATS), utilizing Natural Language Processing (NLP) algorithms, aims to create concise and accurate summaries, thereby significantly reducing the human effort required in processing large volumes of text. ATS has drawn considerable interest in both academic and industrial circles. Many studies have been conducted in the past to survey ATS methods; however, they generally lack practicality for real-world implementations, as they often categorize previous methods from a theoretical standpoint. Moreover, the advent of Large Language Models (LLMs) has altered conventional ATS methods. In this survey, we aim to 1) provide a comprehensive overview of ATS from a ``Process-Oriented Schema'' perspective, which is best aligned with real-world implementations; 2) comprehensively review the latest LLM-based ATS works; and 3) deliver an up-to-date survey of ATS, bridging the two-year gap in the literature. To the best of our knowledge, this is the first survey to specifically investigate LLM-based ATS methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03865v2">Unlocking Reasoning Capabilities in LLMs via Reinforcement Learning Exploration</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has recently enhanced the reasoning capabilities of large language models (LLMs), particularly for mathematical problem solving. However, a fundamental limitation remains: as the sampling budget increases, the advantage of RLVR-trained models over their pretrained bases often diminishes or even vanishes, revealing a strong dependence on the base model's restricted search space. We attribute this phenomenon to the widespread use of the reverse Kullback-Leibler (KL) divergence regularizer, whose mode-seeking behavior keeps the policy trapped inside the base model's support region and hampers wider exploration. To address this issue, we propose RAPO (Rewards-Aware Policy Optimization), an algorithm to promote broader yet focused exploration. Our method (i) utilizes the forward KL penalty to replace the reverse KL penalty for out-of-distribution exploration, and (ii) reweights the reference policy to facilitate adaptive in-distribution exploration. We train Qwen2.5-3B and 7B models with RAPO on the 8K SimpleRL-Zero dataset, without supervised fine-tuning, and evaluate them on AIME2024 and AIME2025. Results show that RAPO consistently improves problem-solving performance. Notably, RAPO enables models to surpass the base model's performance ceiling and solves previously intractable problems, advancing the frontier of RLVR for challenging reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27206v1">Fints: Efficient Inference-Time Personalization for LLMs with Fine-Grained Instance-Tailored Steering</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has intensified the demand for effective personalization techniques that can adapt model behavior to individual user preferences. Despite the non-parametric methods utilizing the in-context learning ability of LLMs, recent parametric adaptation methods, including personalized parameter-efficient fine-tuning and reward modeling emerge. However, these methods face limitations in handling dynamic user patterns and high data sparsity scenarios, due to low adaptability and data efficiency. To address these challenges, we propose a fine-grained and instance-tailored steering framework that dynamically generates sample-level interference vectors from user data and injects them into the model's forward pass for personalized adaptation. Our approach introduces two key technical innovations: a fine-grained steering component that captures nuanced signals by hooking activations from attention and MLP layers, and an input-aware aggregation module that synthesizes these signals into contextually relevant enhancements. The method demonstrates high flexibility and data efficiency, excelling in fast-changing distribution and high data sparsity scenarios. In addition, the proposed method is orthogonal to existing methods and operates as a plug-in component compatible with different personalization techniques. Extensive experiments across diverse scenarios--including short-to-long text generation, and web function calling--validate the effectiveness and compatibility of our approach. Results show that our method significantly enhances personalization performance in fast-shifting environments while maintaining robustness across varying interaction modes and context lengths. Implementation is available at https://github.com/KounianhuaDu/Fints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16559v3">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It contributes to the community in four aspects: (1) a highly customizable benchmarking framework for in-depth comparison and analysis of LLMs; (2) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (3) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions; (4) a baseline LLM agentic workflow that effectively evaluates diverse model capabilities. On eight frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation. The project page is at https://build-arena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18586v2">Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in complex multi-agent applications that use external function calls. This workload creates severe performance challenges for the KV Cache: space contention leads to the eviction of critical agents' caches and time underutilization leaves the cache of agents stalled on long-running tool calls idling in GPU memory. We present Tokencake, a KV-Cache-centric serving framework that co-optimizes scheduling and memory management with an agent-aware design. Tokencake's Space Scheduler uses dynamic memory partitioning to shield critical agents from contention, while its Time Scheduler employs a proactive offload and predictive upload mechanism to repurpose GPU memory during function call stalls. Our evaluation on representative multi-agent benchmarks shows that Tokencake can reduce end-to-end latency by over 47.06%, improve effective GPU memory utilization by up to 16.9% compared to vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27140v1">Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software development, enabling AI-powered applications known as LLM-based agents that promise to automate tasks across diverse apps and workflows. Yet, the security implications of deploying such agents in adversarial mobile environments remain poorly understood. In this paper, we present the first systematic study of security risks in mobile LLM agents. We design and evaluate a suite of adversarial case studies, ranging from opportunistic manipulations such as pop-up advertisements to advanced, end-to-end workflows involving malware installation and cross-app data exfiltration. Our evaluation covers eight state-of-the-art mobile agents across three architectures, with over 2,000 adversarial and paired benign trials. The results reveal systemic vulnerabilities: low-barrier vectors such as fraudulent ads succeed with over 80% reliability, while even workflows requiring the circumvention of operating-system warnings, such as malware installation, are consistently completed by advanced multi-app agents. By mapping these attacks to the MITRE ATT&CK Mobile framework, we uncover novel privilege-escalation and persistence pathways unique to LLM-driven automation. Collectively, our findings provide the first end-to-end evidence that mobile LLM agents are exploitable in realistic adversarial settings, where untrusted third-party channels (e.g., ads, embedded webviews, cross-app notifications) are an inherent part of the mobile ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25761v2">DiagramEval: Evaluating LLM-Generated Diagrams via Graphs</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Diagrams play a central role in research papers for conveying ideas, yet they are often notoriously complex and labor-intensive to create. Although diagrams are presented as images, standard image generative models struggle to produce clear diagrams with well-defined structure. We argue that a promising direction is to generate demonstration diagrams directly in textual form as SVGs, which can leverage recent advances in large language models (LLMs). However, due to the complexity of components and the multimodal nature of diagrams, sufficiently discriminative and explainable metrics for evaluating the quality of LLM-generated diagrams remain lacking. In this paper, we propose DiagramEval, a novel evaluation metric designed to assess demonstration diagrams generated by LLMs. Specifically, DiagramEval conceptualizes diagrams as graphs, treating text elements as nodes and their connections as directed edges, and evaluates diagram quality using two new groups of metrics: node alignment and path alignment. For the first time, we effectively evaluate diagrams produced by state-of-the-art LLMs on recent research literature, quantitatively demonstrating the validity of our metrics. Furthermore, we show how the enhanced explainability of our proposed metrics offers valuable insights into the characteristics of LLM-generated diagrams. Code: https://github.com/ulab-uiuc/diagram-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24046v2">PartnerMAS: An LLM Hierarchical Multi-Agent Framework for Business Partner Selection on High-Dimensional Features</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      High-dimensional decision-making tasks, such as business partner selection, involve evaluating large candidate pools with heterogeneous numerical, categorical, and textual features. While large language models (LLMs) offer strong in-context reasoning capabilities, single-agent or debate-style systems often struggle with scalability and consistency in such settings. We propose PartnerMAS, a hierarchical multi-agent framework that decomposes evaluation into three layers: a Planner Agent that designs strategies, Specialized Agents that perform role-specific assessments, and a Supervisor Agent that integrates their outputs. To support systematic evaluation, we also introduce a curated benchmark dataset of venture capital co-investments, featuring diverse firm attributes and ground-truth syndicates. Across 140 cases, PartnerMAS consistently outperforms single-agent and debate-based multi-agent baselines, achieving up to 10--15\% higher match rates. Analysis of agent reasoning shows that planners are most responsive to domain-informed prompts, specialists produce complementary feature coverage, and supervisors play an important role in aggregation. Our findings demonstrate that structured collaboration among LLM agents can generate more robust outcomes than scaling individual models, highlighting PartnerMAS as a promising framework for high-dimensional decision-making in data-rich domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27119v1">Unstructured Data Analysis using LLMs: A Comprehensive Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Nowadays, the explosion of unstructured data presents immense analytical value. Leveraging the remarkable capability of large language models (LLMs) in extracting attributes of structured tables from unstructured data, researchers are developing LLM-powered data systems for users to analyze unstructured documents as working with a database. These unstructured data analysis (UDA) systems differ significantly in all aspects, including query interfaces, query optimization strategies, and operator implementations, making it unclear which performs best in which scenario. Unfortunately, there does not exist a comprehensive benchmark that offers high-quality, large-volume, and diverse datasets as well as rich query workload to thoroughly evaluate such systems. To fill this gap, we present UDA-Bench, the first benchmark for unstructured data analysis that meets all the above requirements. Specifically, we organize a team with 30 graduate students that spends over in total 10,000 hours on curating 5 datasets from various domains and constructing a relational database view from these datasets by manual annotation. These relational databases can be used as ground truth to evaluate any of these UDA systems despite their differences in programming interfaces. Moreover, we design diverse queries to analyze the attributes defined in the database schema, covering different types of analytical operators with varying selectivities and complexities. We conduct in-depth analysis of the key building blocks of existing UDA systems: query interface, query optimization, operator design, and data processing. We run exhaustive experiments over the benchmark to fully evaluate these systems and different techniques w.r.t. the above building blocks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27107v1">A Memory-Efficient Retrieval Architecture for RAG-Enabled Wearable Medical LLMs-Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 Accepted by BioCAS2025
    </div>
    <details class="paper-abstract">
      With powerful and integrative large language models (LLMs), medical AI agents have demonstrated unique advantages in providing personalized medical consultations, continuous health monitoring, and precise treatment plans. Retrieval-Augmented Generation (RAG) integrates personal medical documents into LLMs by an external retrievable database to address the costly retraining or fine-tuning issues in deploying customized agents. While deploying medical agents in edge devices ensures privacy protection, RAG implementations impose substantial memory access and energy consumption during the retrieval stage. This paper presents a hierarchical retrieval architecture for edge RAG, leveraging a two-stage retrieval scheme that combines approximate retrieval for candidate set generation, followed by high-precision retrieval on pre-selected document embeddings. The proposed architecture significantly reduces energy consumption and external memory access while maintaining retrieval accuracy. Simulation results show that, under TSMC 28nm technology, the proposed hierarchical retrieval architecture has reduced the overall memory access by nearly 50% and the computation by 75% compared to pure INT8 retrieval, and the total energy consumption for 1 MB data retrieval is 177.76 {\mu}J/query.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27106v1">Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      As Natural Language Generation (NLG) continues to be widely adopted, properly assessing it has become quite difficult. Lately, using large language models (LLMs) for evaluating these generations has gained traction, as they tend to align more closely with human preferences than conventional n-gram or embedding-based metrics. In our experiments, we show that LLM judges have low intra-rater reliability in their assigned scores across different runs. This variance makes their ratings inconsistent, almost arbitrary in the worst case, making it difficult to measure how good their judgments actually are. We quantify this inconsistency across different NLG tasks and benchmarks and see if judicious use of LLM judges can still be useful following proper guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16932v2">Prompt-MII: Meta-Learning Instruction Induction for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      A popular method to adapt large language models (LLMs) to new tasks is in-context learning (ICL), which is effective but incurs high inference costs as context length grows. In this paper we propose a method to perform instruction induction, where we take training examples and reduce them to a compact but descriptive prompt that can achieve performance comparable to ICL over the full training set. Specifically, we propose PROMPT-MII, a reinforcement learning (RL) based framework to meta-learn an instruction induction model that can generate compact instructions on the fly for an arbitrary new dataset. We train on over 3,000 diverse classification datasets from the HuggingFace hub, and evaluate on 90 unseen tasks. PROMPT-MII improves downstream model quality by 4-9 F1 points (10-20% relative), matching ICL performance while requiring 3-13x fewer tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25977v2">NeuronMM: High-Performance Matrix Multiplication for LLM Inference on AWS Trainium</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 12 pages, 8 figures, submitted to the Proceedings of the Twenty-First European Conference on Computer Systems (EuroSys'26)
    </div>
    <details class="paper-abstract">
      AI accelerators, customized to AI workloads, provide cost-effective and high-performance solutions for training and inference. Trainium, an AI accelerator recently developed by Amazon Web Services (AWS), provides an attractive option for LLM training and inference through its heterogeneous architecture. However, leveraging Trainium architecture for high performance can be challenging because of its systolic array architecture and special requirement on data layout. In this paper, we design high-performance matrix multiplication (matmul), a critical compute kernel, for LLM inference on Trainium. We introduce a series of techniques customized to Trainium based on kernel fusion and novel caching strategies to reduce data movement across the software-managed memory hierarchy, maximize SRAM bandwidth, and avoid expensive matrix transpose. Evaluating with nine datasets and four recent LLMs, we show that our system largely outperforms the state-of-the-art matmul implemented by AWS on Trainium: at the level of matmul kernel, it achieves an average 1.35x speedup (up to 2.22x), which translates to an average 1.66x speedup (up to 2.49x) for end-to-end LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27072v1">Towards Understanding Self-play for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-31
    </div>
    <details class="paper-abstract">
      Recent advances in large language model (LLM) reasoning, led by reinforcement learning with verifiable rewards (RLVR), have inspired self-play post-training, where models improve by generating and solving their own problems. While self-play has shown strong in-domain and out-of-domain gains, the mechanisms behind these improvements remain poorly understood. In this work, we analyze the training dynamics of self-play through the lens of the Absolute Zero Reasoner, comparing it against RLVR and supervised fine-tuning (SFT). Our study examines parameter update sparsity, entropy dynamics of token distributions, and alternative proposer reward functions. We further connect these dynamics to reasoning performance using pass@k evaluations. Together, our findings clarify how self-play differs from other post-training strategies, highlight its inherent limitations, and point toward future directions for improving LLM math reasoning through self-play.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22023v2">Multimodal Item Scoring for Natural Language Recommendation via Gaussian Process Regression with LLM Relevance Judgments</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 16 pages,20 figures
    </div>
    <details class="paper-abstract">
      Natural Language Recommendation (NLRec) generates item suggestions based on the relevance between user-issued NL requests and NL item description passages. Existing NLRec approaches often use Dense Retrieval (DR) to compute item relevance scores from aggregation of inner products between user request embeddings and relevant passage embeddings. However, DR views the request as the sole relevance label, thus leading to a unimodal scoring function centered on the query embedding that is often a weak proxy for query relevance. To better capture the potential multimodal distribution of the relevance scoring function that may arise from complex NLRec data, we propose GPR-LLM that uses Gaussian Process Regression (GPR) with LLM relevance judgments for a subset of candidate passages. Experiments on four NLRec datasets and two LLM backbones demonstrate that GPR-LLM with an RBF kernel, capable of modeling multimodal relevance scoring functions, consistently outperforms simpler unimodal kernels (dot product, cosine similarity), as well as baseline methods including DR, cross-encoder, and pointwise LLM-based relevance scoring by up to 65%. Overall, GPR-LLM provides an efficient and effective approach to NLRec within a minimal LLM labeling budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18771v3">CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Existing LLM-as-a-Judge approaches for evaluating text generation suffer from rating inconsistencies, with low agreement and high rating variance across different evaluator models. We attribute this to subjective evaluation criteria combined with Likert scale scoring in existing protocols. To address this issue, we introduce CheckEval, a checklist-based evaluation framework that improves rating reliability via decomposed binary questions. Through experiments with 12 evaluator models across multiple datasets, we first demonstrate that CheckEval strongly correlates with human judgments. More importantly, CheckEval dramatically improves the average agreement across evaluator models by 0.45 and reduces the score variance. CheckEval scores furthermore have the benefit of being more interpretable because it decomposes evaluation criteria into traceable binary decisions, allowing analyses of specific attributes driving quality judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02091v3">Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 ICASSP 2025
    </div>
    <details class="paper-abstract">
      Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15413v2">A Comprehensive Evaluation of Cognitive Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 Published in "Proceedings of the 5th International Conference on Natural Language Processing for Digital Humanities"
    </div>
    <details class="paper-abstract">
      We present a large-scale evaluation of 30 cognitive biases in 20 state-of-the-art large language models (LLMs) under various decision-making scenarios. Our contributions include a novel general-purpose test framework for reliable and large-scale generation of tests for LLMs, a benchmark dataset with 30,000 tests for detecting cognitive biases in LLMs, and a comprehensive assessment of the biases found in the 20 evaluated LLMs. Our work confirms and broadens previous findings suggesting the presence of cognitive biases in LLMs by reporting evidence of all 30 tested biases in at least some of the 20 LLMs. We publish our framework code to encourage future research on biases in LLMs: https://github.com/simonmalberg/cognitive-biases-in-llms
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03846v2">Do LLM Evaluators Prefer Themselves for a Reason?</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as automatic evaluators in applications like benchmarking, reward modeling, and self-refinement. Prior work highlights a potential self-preference bias where LLMs favor their own generated responses, a tendency often intensifying with model size and capability. This raises a critical question: Is self-preference harmful, or does it simply reflect the genuinely higher-quality outputs of stronger models? Answering this has been difficult as previous studies relied primarily on subjective tasks. These tasks lack an objective ground truth, meaning that either preference can be reasonably justified. To address this ambiguity, we investigate self-preference using verifiable benchmarks (mathematical reasoning, factual knowledge, code generation) that allow objective ground-truth assessment. This enables us to distinguish harmful self-preference (favoring objectively worse responses) from legitimate self-preference (favoring genuinely superior ones). We conduct large-scale experiments under controlled evaluation conditions across diverse model families (e.g., Llama, Qwen, Gemma, Mistral, Phi, GPT, DeepSeek). Our findings reveal three key insights: (1) While stronger models exhibit greater self-preference, much of this preference aligns with objectively superior performance, indicating stronger models prefer themselves mostly legitimately. (2) Harmful self-preference persists when evaluator models err as generators, and stronger models display more pronounced harmful self-preference when they do err. This suggests stronger models struggle more to recognize when they are wrong. (3) Inference-time scaling strategies, such as generating a long Chain-of-Thought before evaluation, effectively reduce harmful self-preference. These results provide a more nuanced understanding of LLM-based evaluation and practical insights for improving its reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25979v2">AttnCache: Accelerating Self-Attention Inference for LLM Prefill via Attention Cache</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 10 pages, 6 figures, submitted to Ninth Annual Conference on Machine Learning and Systems (MLSys'26)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in generative applications such as chatting, code generation, and reasoning. However, many realworld workloads such as classification, question answering, recommendation, and text embedding rely solely on the prefill stage of inference, where the model encodes input sequences without performing autoregressive decoding. In these prefill only scenarios, the self-attention computation becomes the primary performance bottleneck due to its quadratic complexity with respect to sequence length. In this paper, we observe that semantically different sentences often produce similar attention maps across layers and heads. Building on this insight, we propose AttnCache, a framework that accelerates the prefill stage of LLM inference by retrieving and reusing similar attention maps. Based on an attention map memorization database, AttnCache employs efficient caching and similarity search techniques to identify and reuse pre-cached attention maps during inference, thereby reducing the computational overhead of self-attention. Experimental results show that AttnCache achieves an average of 1.2x end-to-end and 2x attention speedup on CPU, and 1.6x end-to-end and 3x attention speedup on GPU, with negligible accuracy degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16093v2">Beyond Pointwise Scores: Decomposed Criteria-Based Evaluation of LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-10-31
      | 💬 Accepted by 2025 EMNLP industry track
    </div>
    <details class="paper-abstract">
      Evaluating long-form answers in high-stakes domains such as law or medicine remains a fundamental challenge. Standard metrics like BLEU and ROUGE fail to capture semantic correctness, and current LLM-based evaluators often reduce nuanced aspects of answer quality into a single undifferentiated score. We introduce DeCE, a decomposed LLM evaluation framework that separates precision (factual accuracy and relevance) and recall (coverage of required concepts), using instance-specific criteria automatically extracted from gold answer requirements. DeCE is model-agnostic and domain-general, requiring no predefined taxonomies or handcrafted rubrics. We instantiate DeCE to evaluate different LLMs on a real-world legal QA task involving multi-jurisdictional reasoning and citation grounding. DeCE achieves substantially stronger correlation with expert judgments ($r=0.78$), compared to traditional metrics ($r=0.12$), pointwise LLM scoring ($r=0.35$), and modern multidimensional evaluators ($r=0.48$). It also reveals interpretable trade-offs: generalist models favor recall, while specialized models favor precision. Importantly, only 11.95% of LLM-generated criteria required expert revision, underscoring DeCE's scalability. DeCE offers an interpretable and actionable LLM evaluation framework in expert domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26784v1">LLMs Process Lists With General Filter Heads</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Code and data at https://filter.baulab.info/
    </div>
    <details class="paper-abstract">
      We investigate the mechanisms underlying a range of list-processing tasks in LLMs, and we find that LLMs have learned to encode a compact, causal representation of a general filtering operation that mirrors the generic "filter" function of functional programming. Using causal mediation analysis on a diverse set of list-processing tasks, we find that a small number of attention heads, which we dub filter heads, encode a compact representation of the filtering predicate in their query states at certain tokens. We demonstrate that this predicate representation is general and portable: it can be extracted and reapplied to execute the same filtering operation on different collections, presented in different formats, languages, or even in tasks. However, we also identify situations where transformer LMs can exploit a different strategy for filtering: eagerly evaluating if an item satisfies the predicate and storing this intermediate result as a flag directly in the item representations. Our results reveal that transformer LMs can develop human-interpretable implementations of abstract computational operations that generalize in ways that are surprisingly similar to strategies used in traditional functional programming patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09391v2">Comparing human and LLM politeness strategies in free production</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 25 pages, 5 figures | EMNLP 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Polite speech poses a fundamental alignment challenge for large language models (LLMs). Humans deploy a rich repertoire of linguistic strategies to balance informational and social goals -- from positive approaches that build rapport (compliments, expressions of interest) to negative strategies that minimize imposition (hedging, indirectness). We investigate whether LLMs employ a similarly context-sensitive repertoire by comparing human and LLM responses in both constrained and open-ended production tasks. We find that larger models ($\ge$70B parameters) successfully replicate key preferences from the computational pragmatics literature, and human evaluators surprisingly prefer LLM-generated responses in open-ended contexts. However, further linguistic analyses reveal that models disproportionately rely on negative politeness strategies even in positive contexts, potentially leading to misinterpretations. While modern LLMs demonstrate an impressive handle on politeness strategies, these subtle differences raise important questions about pragmatic alignment in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09205v3">Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 5 pages, 5 figures, 2 tables. Accepted at EUSIPCO 2025
    </div>
    <details class="paper-abstract">
      Integrating audio and visual data for training multimodal foundational models remains a challenge. The Audio-Video Vector Alignment (AVVA) framework addresses this by considering AV scene alignment beyond mere temporal synchronization, and leveraging Large Language Models (LLMs) for data curation. AVVA implements a scoring mechanism for selecting aligned training data segments. It integrates Whisper, a speech-based foundation model, for audio and DINOv2 for video analysis in a dual-encoder structure with contrastive learning on AV pairs. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate the effectiveness of the proposed model architecture and data curation approach. AVVA achieves a significant improvement in top-k accuracies for video-to-audio retrieval on all datasets compared to DenseAV, while using only 192 hrs of curated training data. Furthermore, an ablation study indicates that the data curation process effectively trades data quality for data quantity, yielding increases in top-k retrieval accuracies on AudioCaps, VALOR, and VGGSound, compared to training on the full spectrum of uncurated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14681v2">Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted to EMNLP 2025 (Main Conference). Models and evaluation results available at: https://github.com/llm-jp/massive-sft
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a critical step in aligning large language models (LLMs) with human instructions and values, yet many aspects of SFT remain poorly understood. We trained a wide range of base models on a variety of datasets including code generation, mathematical reasoning, and general-domain tasks, resulting in 1,000+ SFT models under controlled conditions. We then identified the dataset properties that matter most and examined the layer-wise modifications introduced by SFT. Our findings reveal that some training-task synergies persist across all models while others vary substantially, emphasizing the importance of model-specific strategies. Moreover, we demonstrate that perplexity consistently predicts SFT effectiveness, often surpassing superficial similarity between the training data and the benchmark, and that mid-layer weight changes correlate most strongly with performance gains. We release these 1,000+ SFT models and benchmark results to accelerate further research. All resources are available at https://github.com/llm-jp/massive-sft.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26707v1">Value Drifts: Tracing Value Alignment During LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      As LLMs occupy an increasingly important role in society, they are more and more confronted with questions that require them not only to draw on their general knowledge but also to align with certain human value systems. Therefore, studying the alignment of LLMs with human values has become a crucial field of inquiry. Prior work, however, mostly focuses on evaluating the alignment of fully trained models, overlooking the training dynamics by which models learn to express human values. In this work, we investigate how and at which stage value alignment arises during the course of a model's post-training. Our analysis disentangles the effects of post-training algorithms and datasets, measuring both the magnitude and time of value drifts during training. Experimenting with Llama-3 and Qwen-3 models of different sizes and popular supervised fine-tuning (SFT) and preference optimization datasets and algorithms, we find that the SFT phase generally establishes a model's values, and subsequent preference optimization rarely re-aligns these values. Furthermore, using a synthetic preference dataset that enables controlled manipulation of values, we find that different preference optimization algorithms lead to different value alignment outcomes, even when preference data is held constant. Our findings provide actionable insights into how values are learned during post-training and help to inform data curation, as well as the selection of models and algorithms for preference optimization to improve model alignment to human values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23254v3">MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 16 pages, 21 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Owing to the huge success of generative artificial intelligence (AI), large language models (LLMs) have emerged as a core subclass, underpinning applications such as question answering, text generation, and code completion. While fine-tuning these models on domain-specific data can yield significant performance gains, it also poses daunting computational challenges, especially for researchers and small organizations with limited hardware resources. Although SSD offloading (i.e., ZeRO-Infinity) has emerged as a viable strategy to overcome the GPU memory barrier via leveraging both system memory (i.e., CPU DRAM) and storage space (i.e., solid-state devices, SSDs), its design primarily targets model-centric performance issues. As a result, key system-level issues, including system memory fragmentation, inefficient pinned buffer allocation, peak CPU usage spikes, and file system overhead, remain unaddressed, stifling scalability and inflating costs. Such an observation motivates this paper to introduce MemAscend, a framework that systematically tackles the underexplored system memory bottlenecks in SSD-offloaded LLM training, with a focus on resource-constrained environments. By streamlining pinned-memory allocation, eradicating fragmentation, and mitigating peak overhead, MemAscend reclaims a substantial system memory budget, enabling larger models, longer context windows, and higher batch sizes without exceeding modest hardware limits. Across diverse LLM benchmarks, MemAscend reduces peak system-memory consumption by an average of 55.7% compared with standard SSD offloading techniques, lowering the hardware barrier for fine-tuning and unlocking new possibilities for cost-effective large-scale training on limited-resource machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03305v2">Analysis and Optimized CXL-Attached Memory Allocation for Long-Context LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 13 pages, 15 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The substantial memory requirements of Large Language Models (LLMs), particularly for long-context fine-tuning, have renewed interest in CPU offloading to augment limited GPU memory. However, as context lengths grow, relying on CPU memory for intermediate states introduces a significant bottleneck that can exhaust the capacity of mainstream client platforms. To address this limitation, this work investigates the effectiveness of Compute Express Link (CXL) add-in card (AIC) memory as an extension to CPU memory, enabling larger model sizes and longer context lengths during fine-tuning. Extensive benchmarking reveals two critical challenges. First, current deep learning frameworks such as PyTorch lack fine-grained, per-tensor control over NUMA memory allocation, exposing only coarse, process-level policies. Second, due to this lack of control, when the memory footprint of fine-tuning is offloaded across local DRAM and CXL-attached memory, naively placing optimizer data in higher-latency CXL leads to substantial slowdowns in the optimizer step (e.g., 4x once data exceeds 20M elements). To overcome these challenges, this work introduces a PyTorch extension that enables tensor-level system memory control and a CXL-aware memory allocator that pins latency-critical tensors in local DRAM while maximizing bandwidth by striping latency-tolerant tensors across one or more CXL devices. Evaluated on a real hardware setup with 7B and 12B models, 4K-32K contexts, and a single GPU, our approach recovers throughput to 97-99% of DRAM-only with a single AIC and approximately 100% with two AICs, delivering up to 21% improvement over naive interleaving while preserving DRAM-like DMA bandwidth for GPU transfers. These results show that carefully managed CXL-attached memory is a practical path to scaling long-context fine-tuning beyond DRAM limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01543v2">Refine-n-Judge: Curating High-Quality Preference Chains for LLM-Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable progress through preference-based fine-tuning, which critically depends on the quality of the underlying training data. While human feedback is essential for improving data quality, it is costly and does not scale well. In this paper, we introduce Refine-n-Judge, an automated iterative approach that leverages a single LLM as both a refiner and a judge to enhance dataset quality. Unlike existing iterative refinement methods, Refine-n-Judge employs an LLM to both generate refinements and explicitly evaluate each improvement, ensuring that every iteration meaningfully enhances the dataset without requiring additional human annotation or a separate reward model. At each step, the LLM refines a response and judges whether the refinement is an improvement over the previous answer. This process continues until the LLM prefers the initial answer over the refinement, indicating no further improvements. This produces sequences of increasing quality, preference-labeled responses ideal for fine-tuning. We demonstrate the effectiveness of Refine-n-Judge across a range of public datasets spanning five corpora, targeting tasks such as coding, math, and conversation. Models (Llama 3.1-8B and Llama 3.3-70B) fine-tuned on Refine-n-Judge-enhanced datasets were preferred by LLM judges in over 74% of comparisons against models tuned on the original dataset by GPT-4. Additionally, we report performance gains: +5% on AlpacaEval and AlpacaEval 2.0, and +19% on MT-Bench. Our results indicate that Refine-n-Judge produces high-quality datasets and scalable model improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21257v2">CompoST: A Benchmark for Analyzing the Ability of LLMs To Compositionally Interpret Questions in a QALD Setting</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Research Track, 24th International Semantic Web Conference (ISWC 2025), November 2-6, 2025, Nara, Japan
    </div>
    <details class="paper-abstract">
      Language interpretation is a compositional process, in which the meaning of more complex linguistic structures is inferred from the meaning of their parts. Large language models possess remarkable language interpretation capabilities and have been successfully applied to interpret questions by mapping them to SPARQL queries. An open question is how systematic this interpretation process is. Toward this question, in this paper, we propose a benchmark for investigating to what extent the abilities of LLMs to interpret questions are actually compositional. For this, we generate three datasets of varying difficulty based on graph patterns in DBpedia, relying on Lemon lexica for verbalization. Our datasets are created in a very controlled fashion in order to test the ability of LLMs to interpret structurally complex questions, given that they have seen the atomic building blocks. This allows us to evaluate to what degree LLMs are able to interpret complex questions for which they "understand" the atomic parts. We conduct experiments with models of different sizes using both various prompt and few-shot optimization techniques as well as fine-tuning. Our results show that performance in terms of macro $F_1$ degrades from $0.45$ over $0.26$ down to $0.09$ with increasing deviation from the samples optimized on. Even when all necessary information was provided to the model in the input, the $F_1$ scores do not exceed $0.57$ for the dataset of lowest complexity. We thus conclude that LLMs struggle to systematically and compositionally interpret questions and map them into SPARQL queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26641v1">All You Need for Object Detection: From Pixels, Points, and Prompts to Next-Gen Fusion and Multimodal LLMs/VLMs in Autonomous Vehicles</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Autonomous Vehicles (AVs) are transforming the future of transportation through advances in intelligent perception, decision-making, and control systems. However, their success is tied to one core capability, reliable object detection in complex and multimodal environments. While recent breakthroughs in Computer Vision (CV) and Artificial Intelligence (AI) have driven remarkable progress, the field still faces a critical challenge as knowledge remains fragmented across multimodal perception, contextual reasoning, and cooperative intelligence. This survey bridges that gap by delivering a forward-looking analysis of object detection in AVs, emphasizing emerging paradigms such as Vision-Language Models (VLMs), Large Language Models (LLMs), and Generative AI rather than re-examining outdated techniques. We begin by systematically reviewing the fundamental spectrum of AV sensors (camera, ultrasonic, LiDAR, and Radar) and their fusion strategies, highlighting not only their capabilities and limitations in dynamic driving environments but also their potential to integrate with recent advances in LLM/VLM-driven perception frameworks. Next, we introduce a structured categorization of AV datasets that moves beyond simple collections, positioning ego-vehicle, infrastructure-based, and cooperative datasets (e.g., V2V, V2I, V2X, I2I), followed by a cross-analysis of data structures and characteristics. Ultimately, we analyze cutting-edge detection methodologies, ranging from 2D and 3D pipelines to hybrid sensor fusion, with particular attention to emerging transformer-driven approaches powered by Vision Transformers (ViTs), Large and Small Language Models (SLMs), and VLMs. By synthesizing these perspectives, our survey delivers a clear roadmap of current capabilities, open challenges, and future opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17197v2">SignalLLM: A General-Purpose LLM Agent Framework for Automated Signal Processing</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Modern signal processing (SP) pipelines, whether model-based or data-driven, often constrained by complex and fragmented workflow, rely heavily on expert knowledge and manual engineering, and struggle with adaptability and generalization under limited data. In contrast, Large Language Models (LLMs) offer strong reasoning capabilities, broad general-purpose knowledge, in-context learning, and cross-modal transfer abilities, positioning them as powerful tools for automating and generalizing SP workflows. Motivated by these potentials, we introduce SignalLLM, the first general-purpose LLM-based agent framework for general SP tasks. Unlike prior LLM-based SP approaches that are limited to narrow applications or tricky prompting, SignalLLM introduces a principled, modular architecture. It decomposes high-level SP goals into structured subtasks via in-context learning and domain-specific retrieval, followed by hierarchical planning through adaptive retrieval-augmented generation (RAG) and refinement; these subtasks are then executed through prompt-based reasoning, cross-modal reasoning, code synthesis, model invocation, or data-driven LLM-assisted modeling. Its generalizable design enables the flexible selection of problem solving strategies across different signal modalities, task types, and data conditions. We demonstrate the versatility and effectiveness of SignalLLM through five representative tasks in communication and sensing, such as radar target detection, human activity recognition, and text compression. Experimental results show superior performance over traditional and existing LLM-based methods, particularly in few-shot and zero-shot settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01369v2">Incentivizing LLMs to Self-Verify Their Answers</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable progress in complex reasoning tasks through both post-training and test-time scaling laws. While prevalent test-time scaling approaches are often realized by using external reward models to guide the model generation process, we find that only marginal gains can be acquired when scaling a model post-trained on specific reasoning tasks. We identify that the limited improvement stems from distribution discrepancies between the specific post-trained generator and the general reward model. To address this, we propose a framework that incentivizes LLMs to self-verify their own answers. By unifying answer generation and verification within a single reinforcement learning (RL) process, we train models that can effectively assess the correctness of their own solutions. The trained model can further scale its performance at inference time by verifying its generations, without the need for external verifiers. We train our self-verification models based on Qwen2.5-Math-7B and DeepSeek-R1-Distill-Qwen-1.5B, demonstrating their capabilities across varying reasoning context lengths. Experiments on multiple mathematical reasoning benchmarks show that our models can not only improve post-training performance but also enable effective test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26546v1">WeaveRec: An LLM-Based Cross-Domain Sequential Recommendation Framework with Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Cross-Domain Sequential Recommendation (CDSR) seeks to improve user preference modeling by transferring knowledge from multiple domains. Despite the progress made in CDSR, most existing methods rely on overlapping users or items to establish cross-domain correlations-a requirement that rarely holds in real-world settings. The advent of large language models (LLM) and model-merging techniques appears to overcome this limitation by unifying multi-domain data without explicit overlaps. Yet, our empirical study shows that naively training an LLM on combined domains-or simply merging several domain-specific LLMs-often degrades performance relative to a model trained solely on the target domain. To address these challenges, we first experimentally investigate the cause of suboptimal performance in LLM-based cross-domain recommendation and model merging. Building on these insights, we introduce WeaveRec, which cross-trains multiple LoRA modules with source and target domain data in a weaving fashion, and fuses them via model merging. WeaveRec can be extended to multi-source domain scenarios and notably does not introduce additional inference-time cost in terms of latency or memory. Furthermore, we provide a theoretical guarantee that WeaveRec can reduce the upper bound of the expected error in the target domain. Extensive experiments on single-source, multi-source, and cross-platform cross-domain recommendation scenarios validate that WeaveRec effectively mitigates performance degradation and consistently outperforms baseline approaches in real-world recommendation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15030v3">Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26510v1">LLMs as In-Context Meta-Learners for Model and Hyperparameter Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 27 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Model and hyperparameter selection are critical but challenging in machine learning, typically requiring expert intuition or expensive automated search. We investigate whether large language models (LLMs) can act as in-context meta-learners for this task. By converting each dataset into interpretable metadata, we prompt an LLM to recommend both model families and hyperparameters. We study two prompting strategies: (1) a zero-shot mode relying solely on pretrained knowledge, and (2) a meta-informed mode augmented with examples of models and their performance on past tasks. Across synthetic and real-world benchmarks, we show that LLMs can exploit dataset metadata to recommend competitive models and hyperparameters without search, and that improvements from meta-informed prompting demonstrate their capacity for in-context meta-learning. These results highlight a promising new role for LLMs as lightweight, general-purpose assistants for model selection and hyperparameter optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08271v2">RecCocktail: A Generalizable and Efficient Framework for LLM-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in recent years, owing to their impressive generalization capabilities and rich world knowledge. To capitalize on the potential of using LLMs as recommender systems, mainstream approaches typically focus on two paradigms. The first paradigm designs multi-domain or multi-task instruction data for generalizable recommendation, so as to align LLMs with general recommendation areas and deal with cold-start recommendation. The second paradigm focuses on enhancing domain-specific recommendation tasks, improving performance in warm recommendation scenarios. While most previous works treat these two paradigms separately, we argue that they have complementary advantages, and combining them can yield better results. In this paper, we propose a generalizable and efficient LLM-based recommendation framework RecCocktail. Our approach begins with fine-tuning a "base spirit" LoRA module using domain-general recommendation instruction data to align LLM with recommendation knowledge. Next, given users' behavior of a specific domain, we construct a domain-specific "ingredient" LoRA module. We then provide an entropy-guided adaptive merging method to mix the "base spirit" and the "ingredient" in the weight space. Please note that, RecCocktail combines the advantages of the existing two paradigms without introducing additional time or space overhead during the inference phase. Moreover, RecCocktail is efficient with plug and play, as the "base spirit" LoRA is trained only once, and any domain-specific "ingredient" can be efficiently mixed with only domain-specific fine-tuning. Extensive experiments on multiple datasets under both warm and cold-start recommendation scenarios validate the effectiveness and generality of the proposed RecCocktail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26494v1">Simulating and Experimenting with Social Media Mobilization Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Online social networks have transformed the ways in which political mobilization messages are disseminated, raising new questions about how peer influence operates at scale. Building on the landmark 61-million-person Facebook experiment \citep{bond201261}, we develop an agent-based simulation framework that integrates real U.S. Census demographic distributions, authentic Twitter network topology, and heterogeneous large language model (LLM) agents to examine the effect of mobilization messages on voter turnout. Each simulated agent is assigned demographic attributes, a personal political stance, and an LLM variant (\texttt{GPT-4.1}, \texttt{GPT-4.1-Mini}, or \texttt{GPT-4.1-Nano}) reflecting its political sophistication. Agents interact over realistic social network structures, receiving personalized feeds and dynamically updating their engagement behaviors and voting intentions. Experimental conditions replicate the informational and social mobilization treatments of the original Facebook study. Across scenarios, the simulator reproduces qualitative patterns observed in field experiments, including stronger mobilization effects under social message treatments and measurable peer spillovers. Our framework provides a controlled, reproducible environment for testing counterfactual designs and sensitivity analyses in political mobilization research, offering a bridge between high-validity field experiments and flexible computational modeling.\footnote{Code and data available at https://github.com/CausalMP/LLM-SocioPol}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26490v1">Scaffolding Creativity: How Divergent and Convergent LLM Personas Shape Human Machine Creative Problem-Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly shaping creative work and problem-solving; however, prior research suggests that they may diminish unassisted creativity. To address this tension, a coach-like LLM environment was developed that embodies divergent and convergent thinking personas as two complementary processes. Effectiveness and user behavior were assessed through a controlled experiment in which participants interacted with either persona, while a control group engaged with a standard LLM providing direct answers. Notably, users' perceptions of which persona best supported their creativity often diverged from objective performance measures. Trait-based analyses revealed that individual differences predict when people utilize divergent versus convergent personas, suggesting opportunities for adaptive sequencing. Furthermore, interaction patterns reflected the design thinking model, demonstrating how persona-guided support shapes creative problem-solving. Our findings provide design principles for creativity support systems that strike a balance between exploration and convergence through persona-based guidance and personalization. These insights advance human-AI collaboration tools that scaffold rather than overshadow human creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26486v1">LINK-KG: LLM-Driven Coreference-Resolved Knowledge Graphs for Human Smuggling Networks</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted in ICKG 2025 Conference, 8 Pages, 2 Figures
    </div>
    <details class="paper-abstract">
      Human smuggling networks are complex and constantly evolving, making them difficult to analyze comprehensively. Legal case documents offer rich factual and procedural insights into these networks but are often long, unstructured, and filled with ambiguous or shifting references, posing significant challenges for automated knowledge graph (KG) construction. Existing methods either overlook coreference resolution or fail to scale beyond short text spans, leading to fragmented graphs and inconsistent entity linking. We propose LINK-KG, a modular framework that integrates a three-stage, LLM-guided coreference resolution pipeline with downstream KG extraction. At the core of our approach is a type-specific Prompt Cache, which consistently tracks and resolves references across document chunks, enabling clean and disambiguated narratives for structured knowledge graph construction from both short and long legal texts. LINK-KG reduces average node duplication by 45.21% and noisy nodes by 32.22% compared to baseline methods, resulting in cleaner and more coherent graph structures. These improvements establish LINK-KG as a strong foundation for analyzing complex criminal networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26480v1">Automated Extract Method Refactoring with Open-Source LLMs: A Comparative Study</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted at AIware'25 - Main Track
    </div>
    <details class="paper-abstract">
      Automating the Extract Method refactoring (EMR) remains challenging and largely manual despite its importance in improving code readability and maintainability. Recent advances in open-source, resource-efficient Large Language Models (LLMs) offer promising new approaches for automating such high-level tasks. In this work, we critically evaluate five state-of-the-art open-source LLMs, spanning 3B to 8B parameter sizes, on the EMR task for Python code. We systematically assess functional correctness and code quality using automated metrics and investigate the impact of prompting strategies by comparing one-shot prompting to a Recursive criticism and improvement (RCI) approach. RCI-based prompting consistently outperforms one-shot prompting in test pass rates and refactoring quality. The best-performing models, Deepseek-Coder-RCI and Qwen2.5-Coder-RCI, achieve test pass percentage (TPP) scores of 0.829 and 0.808, while reducing lines of code (LOC) per method from 12.103 to 6.192 and 5.577, and cyclomatic complexity (CC) from 4.602 to 3.453 and 3.294, respectively. A developer survey on RCI-generated refactorings shows over 70% acceptance, with Qwen2.5-Coder rated highest across all evaluation criteria. In contrast, the original code scored below neutral, particularly in readability and maintainability, underscoring the benefits of automated refactoring guided by quality prompts. While traditional metrics like CC and LOC provide useful signals, they often diverge from human judgments, emphasizing the need for human-in-the-loop evaluation. Our open-source benchmark offers a foundation for future research on automated refactoring with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02091v2">Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 ICASSP 2025
    </div>
    <details class="paper-abstract">
      Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21513v2">Wisdom and Delusion of LLM Ensembles for Code Generation and Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Added Acknowledgments section and hyphenated last names
    </div>
    <details class="paper-abstract">
      Today's pursuit of a single Large Language Model (LMM) for all software engineering tasks is resource-intensive and overlooks the potential benefits of complementarity, where different models contribute unique strengths. However, the degree to which coding LLMs complement each other and the best strategy for maximizing an ensemble's potential are unclear, leaving practitioners without a clear path to move beyond single-model systems. To address this gap, we empirically compare ten individual LLMs from five families, and three ensembles of these LLMs across three software engineering benchmarks covering code generation and program repair. We assess the complementarity between models and the performance gap between the best individual model and the ensembles. Next, we evaluate various selection heuristics to identify correct solutions from an ensemble's candidate pool. We find that the theoretical upperbound for an ensemble's performance can be 83% above the best single model. Our results show that consensus-based strategies for selecting solutions fall into a "popularity trap," amplifying common but incorrect outputs. In contrast, a diversity-based strategy realizes up to 95% of this theoretical potential, and proves effective even in small two-model ensembles, enabling a cost-efficient way to enhance performance by leveraging multiple LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23127v2">Lost in Tokenization: Context as the Key to Unlocking Biomolecular Understanding in Scientific LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 38 pages, under review
    </div>
    <details class="paper-abstract">
      Scientific Large Language Models (Sci-LLMs) have emerged as a promising frontier for accelerating biological discovery. However, these models face a fundamental challenge when processing raw biomolecular sequences: the tokenization dilemma. Whether treating sequences as a specialized language, risking the loss of functional motif information, or as a separate modality, introducing formidable alignment challenges, current strategies fundamentally limit their reasoning capacity. We challenge this sequence-centric paradigm by positing that a more effective strategy is to provide Sci-LLMs with high-level structured context derived from established bioinformatics tools, thereby bypassing the need to interpret low-level noisy sequence data directly. Through a systematic comparison of leading Sci-LLMs on biological reasoning tasks, we tested three input modes: sequence-only, context-only, and a combination of both. Our findings are striking: the context-only approach consistently and substantially outperforms all other modes. Even more revealing, the inclusion of the raw sequence alongside its high-level context consistently degrades performance, indicating that raw sequences act as informational noise, even for models with specialized tokenization schemes. These results suggest that the primary strength of existing Sci-LLMs lies not in their nascent ability to interpret biomolecular syntax from scratch, but in their profound capacity for reasoning over structured, human-readable knowledge. Therefore, we argue for reframing Sci-LLMs not as sequence decoders, but as powerful reasoning engines over expert knowledge. This work lays the foundation for a new class of hybrid scientific AI agents, repositioning the developmental focus from direct sequence interpretation towards high-level knowledge synthesis. The code is available at https://github.com/opendatalab-raiser/CoKE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11329v4">TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 14 pages, 16 figures. For source code, see https://github.com/microsoft/tokenweave. In version 2, Figure 6 shows All-Reduce bandwidth instead of Reduce-Scatter. The Multimem Reduce-Scatter bandwidth formula differs slightly from the ring-based version. Fixed x-ticks in Figure 7
    </div>
    <details class="paper-abstract">
      Distributed inference of large language models (LLMs) can introduce overheads of up to 20% even over GPUs connected via high-speed interconnects such as NVLink. Multiple techniques have been proposed to mitigate these overheads by decomposing computations into finer-grained tasks and overlapping communication with sub-tasks as they complete. However, fine-grained decomposition of a large computation into many smaller computations on GPUs results in overheads. Furthermore, the communication itself uses many streaming multiprocessors (SMs), adding to the overhead. We present TokenWeave to address these challenges. TokenWeave proposes a Token-Splitting technique that divides the tokens in the inference batch into two approximately equal subsets in a wave-aware manner. The communication of one subset is then overlapped with the computation of the other. In addition, TokenWeave optimizes the order of the layer normalization computation with respect to communication operations and implements a novel fused AllReduce--RMSNorm kernel that carefully leverages Multimem instruction support available on Hopper and Blackwell NVIDIA GPUs. These optimizations allow TokenWeave to perform communication and RMSNorm using only 2-8 SMs. Moreover, our kernel enables the memory-bound RMSNorm to be overlapped with the other batch's computation, providing additional gains. Our evaluations demonstrate up to 1.29x speedup in latency and 1.26x higher throughput across multiple models and workloads. In several settings, TokenWeave results in better performance compared to an equivalent model with all communication removed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25536v2">TwinVoice: A Multi-dimensional Benchmark Towards Digital Twins via LLM Persona Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Main paper: 11 pages, 3 figures, 6 tables. Appendix: 28 pages. Bangde Du and Minghao Guo contributed equally. Corresponding authors: Ziyi Ye (ziyiye@fudan.edu.cn), Qingyao Ai (aiqy@tsinghua.edu.cn)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are exhibiting emergent human-like abilities and are increasingly envisioned as the foundation for simulating an individual's communication style, behavioral tendencies, and personality traits. However, current evaluations of LLM-based persona simulation remain limited: most rely on synthetic dialogues, lack systematic frameworks, and lack analysis of the capability requirement. To address these limitations, we introduce TwinVoice, a comprehensive benchmark for assessing persona simulation across diverse real-world contexts. TwinVoice encompasses three dimensions: Social Persona (public social interactions), Interpersonal Persona (private dialogues), and Narrative Persona (role-based expression). It further decomposes the evaluation of LLM performance into six fundamental capabilities, including opinion consistency, memory recall, logical reasoning, lexical fidelity, persona tone, and syntactic style. Experimental results reveal that while advanced models achieve moderate accuracy in persona simulation, they still fall short of capabilities such as syntactic style and memory recall. Consequently, the average performance achieved by LLMs remains considerably below the human baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26374v1">BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Reinforcement finetuning (RFT) is a key technique for aligning Large Language Models (LLMs) with human preferences and enhancing reasoning, yet its effectiveness is highly sensitive to which tasks are explored during training. Uniform task sampling is inefficient, wasting computation on tasks that are either trivial or unsolvable, while existing task selection methods often suffer from high rollout costs, poor adaptivity, or incomplete evidence. We introduce \textbf{BOTS}, a unified framework for \textbf{B}ayesian \textbf{O}nline \textbf{T}ask \textbf{S}election in LLM reinforcement finetuning. Grounded in Bayesian inference, BOTS adaptively maintains posterior estimates of task difficulty as the model evolves. It jointly incorporates \emph{explicit evidence} from direct evaluations of selected tasks and \emph{implicit evidence} inferred from these evaluations for unselected tasks, with Thompson sampling ensuring a principled balance between exploration and exploitation. To make implicit evidence practical, we instantiate it with an ultra-light interpolation-based plug-in that estimates difficulties of unevaluated tasks without extra rollouts, adding negligible overhead. Empirically, across diverse domains and LLM scales, BOTS consistently improves data efficiency and performance over baselines and ablations, providing a practical and extensible solution for dynamic task selection in RFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26336v1">From Amateur to Master: Infusing Knowledge into LLMs via Automated Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at general tasks but underperform in specialized domains like economics and psychology, which require deep, principled understanding. To address this, we introduce ACER (Automated Curriculum-Enhanced Regimen) that transforms generalist models into domain experts without sacrificing their broad capabilities. ACER first synthesizes a comprehensive, textbook-style curriculum by generating a table of contents for a subject and then creating question-answer (QA) pairs guided by Bloom's taxonomy. This ensures systematic topic coverage and progressively increasing difficulty. The resulting synthetic corpus is used for continual pretraining with an interleaved curriculum schedule, aligning learning across both content and cognitive dimensions. Experiments with Llama 3.2 (1B and 3B) show significant gains in specialized MMLU subsets. In challenging domains like microeconomics, where baselines struggle, ACER boosts accuracy by 5 percentage points. Across all target domains, we observe a consistent macro-average improvement of 3 percentage points. Notably, ACER not only prevents catastrophic forgetting but also facilitates positive cross-domain knowledge transfer, improving performance on non-target domains by 0.7 points. Beyond MMLU, ACER enhances performance on knowledge-intensive benchmarks like ARC and GPQA by over 2 absolute points, while maintaining stable performance on general reasoning tasks. Our results demonstrate that ACER offers a scalable and effective recipe for closing critical domain gaps in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26309v1">GraphCompliance: Aligning Policy and Context Graphs for LLM-Based Regulatory Compliance</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Under review at The Web Conference 2026 (Semantics & Knowledge track). Code will be released upon acceptance. This arXiv v1 contains no repository links to preserve double-blind review
    </div>
    <details class="paper-abstract">
      Compliance at web scale poses practical challenges: each request may require a regulatory assessment. Regulatory texts (e.g., the General Data Protection Regulation, GDPR) are cross-referential and normative, while runtime contexts are expressed in unstructured natural language. This setting motivates us to align semantic information in unstructured text with the structured, normative elements of regulations. To this end, we introduce GraphCompliance, a framework that represents regulatory texts as a Policy Graph and runtime contexts as a Context Graph, and aligns them. In this formulation, the policy graph encodes normative structure and cross-references, whereas the context graph formalizes events as subject-action-object (SAO) and entity-relation triples. This alignment anchors the reasoning of a judge large language model (LLM) in structured information and helps reduce the burden of regulatory interpretation and event parsing, enabling a focus on the core reasoning step. In experiments on 300 GDPR-derived real-world scenarios spanning five evaluation tasks, GraphCompliance yields 4.1-7.2 percentage points (pp) higher micro-F1 than LLM-only and RAG baselines, with fewer under- and over-predictions, resulting in higher recall and lower false positive rates. Ablation studies indicate contributions from each graph component, suggesting that structured representations and a judge LLM are complementary for normative reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04380v3">Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted by NeurIPS'25 main track. 47 pages, 21 figures, 32 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this work, we investigate the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-design for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26277v1">Do LLMs Signal When They're Right? Evidence from Neuron Agreement</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) commonly boost reasoning via sample-evaluate-ensemble decoders, achieving label free gains without ground truth. However, prevailing strategies score candidates using only external outputs such as token probabilities, entropies, or self evaluations, and these signals can be poorly calibrated after post training. We instead analyze internal behavior based on neuron activations and uncover three findings: (1) external signals are low dimensional projections of richer internal dynamics; (2) correct responses activate substantially fewer unique neurons than incorrect ones throughout generation; and (3) activations from correct responses exhibit stronger cross sample agreement, whereas incorrect ones diverge. Motivated by these observations, we propose Neuron Agreement Decoding (NAD), an unsupervised best-of-N method that selects candidates using activation sparsity and cross sample neuron agreement, operating solely on internal signals and without requiring comparable textual outputs. NAD enables early correctness prediction within the first 32 generated tokens and supports aggressive early stopping. Across math and science benchmarks with verifiable answers, NAD matches majority voting; on open ended coding benchmarks where majority voting is inapplicable, NAD consistently outperforms Avg@64. By pruning unpromising trajectories early, NAD reduces token usage by 99% with minimal loss in generation quality, showing that internal signals provide reliable, scalable, and efficient guidance for label free ensemble decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26274v1">PVMark: Enabling Public Verifiability for LLM Watermarking Schemes</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Watermarking schemes for large language models (LLMs) have been proposed to identify the source of the generated text, mitigating the potential threats emerged from model theft. However, current watermarking solutions hardly resolve the trust issue: the non-public watermark detection cannot prove itself faithfully conducting the detection. We observe that it is attributed to the secret key mostly used in the watermark detection -- it cannot be public, or the adversary may launch removal attacks provided the key; nor can it be private, or the watermarking detection is opaque to the public. To resolve the dilemma, we propose PVMark, a plugin based on zero-knowledge proof (ZKP), enabling the watermark detection process to be publicly verifiable by third parties without disclosing any secret key. PVMark hinges upon the proof of `correct execution' of watermark detection on which a set of ZKP constraints are built, including mapping, random number generation, comparison, and summation. We implement multiple variants of PVMark in Python, Rust and Circom, covering combinations of three watermarking schemes, three hash functions, and four ZKP protocols, to show our approach effectively works under a variety of circumstances. By experimental results, PVMark efficiently enables public verifiability on the state-of-the-art LLM watermarking schemes yet without compromising the watermarking performance, promising to be deployed in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26270v1">Graph-Enhanced Policy Optimization in LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Under review as a conference paper
    </div>
    <details class="paper-abstract">
      Group based reinforcement learning (RL) has shown impressive results on complex reasoning and mathematical tasks. Yet, when applied to train multi-turn, interactive LLM agents, these methods often suffer from structural blindness-the inability to exploit the underlying connectivity of the environment. This manifests in three critical challenges: (1) inefficient, unguided exploration, (2) imprecise credit assignment due to overlooking pivotal states, and (3) myopic planning caused by static reward discounting. We address these issues with Graph-Enhanced Policy Optimization (GEPO), which dynamically constructs a state-transition graph from agent experience and employs graph-theoretic centrality to provide three synergistic learning signals: (1)structured intrinsic rewards that guide exploration toward high-impact states, (2) a graph-enhanced advantage function for topology-aware credit assignment, and (3) a dynamic discount factor adapted to each state's strategic value. On the ALFWorld, WebShop, and a proprietary Workbench benchmarks, GEPO demonstrates strong performance, achieving absolute success rate gains of +4.1%, +5.3%, and +10.9% over competitive baselines. These results highlight that explicitly modeling environmental structure is a robust, generalizable strategy for advancing LLM agent training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13229v3">IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26242v1">Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      With increasing urban traffic complexity, Traffic Signal Control (TSC) is essential for optimizing traffic flow and improving road safety. Large Language Models (LLMs) emerge as promising approaches for TSC. However, they are prone to hallucinations in emergencies, leading to unreliable decisions that may cause substantial delays for emergency vehicles. Moreover, diverse intersection types present substantial challenges for traffic state encoding and cross-intersection training, limiting generalization across heterogeneous intersections. Therefore, this paper proposes Retrieval Augmented Generation (RAG)-enhanced distributed LLM agents with Emergency response for Generalizable TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning framework, which dynamically adjusts reasoning depth based on the emergency scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to distill specific knowledge and guidance from historical cases, enhancing the reliability and rationality of agents' emergency decisions. Secondly, this paper designs a type-agnostic traffic representation and proposes a Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3 adaptively samples training experience from diverse intersections with environment feedback-based priority and fine-tunes LLM agents with a designed reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies across heterogeneous intersections. On three real-world road networks with 17 to 177 heterogeneous intersections, extensive experiments show that REG-TSC reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle waiting time by 83.16%, outperforming other state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26238v1">Questionnaire meets LLM: A Benchmark and Empirical Study of Structural Skills for Understanding Questions and Responses</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 14 pages, 3 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Millions of people take surveys every day, from market polls and academic studies to medical questionnaires and customer feedback forms. These datasets capture valuable insights, but their scale and structure present a unique challenge for large language models (LLMs), which otherwise excel at few-shot reasoning over open-ended text. Yet, their ability to process questionnaire data or lists of questions crossed with hundreds of respondent rows remains underexplored. Current retrieval and survey analysis tools (e.g., Qualtrics, SPSS, REDCap) are typically designed for humans in the workflow, limiting such data integration with LLM and AI-empowered automation. This gap leaves scientists, surveyors, and everyday users without evidence-based guidance on how to best represent questionnaires for LLM consumption. We address this by introducing QASU (Questionnaire Analysis and Structural Understanding), a benchmark that probes six structural skills, including answer lookup, respondent count, and multi-hop inference, across six serialization formats and multiple prompt strategies. Experiments on contemporary LLMs show that choosing an effective format and prompt combination can improve accuracy by up to 8.8% points compared to suboptimal formats. For specific tasks, carefully adding a lightweight structural hint through self-augmented prompting can yield further improvements of 3-4% points on average. By systematically isolating format and prompting effects, our open source benchmark offers a simple yet versatile foundation for advancing both research and real-world practice in LLM-based questionnaire analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26219v1">Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit space</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 21 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Test-time alignment of large language models (LLMs) attracts attention because fine-tuning LLMs requires high computational costs. In this paper, we propose a new test-time alignment method called adaptive importance sampling on pre-logits (AISP) on the basis of the sampling-based model predictive control with the stochastic control input. AISP applies the Gaussian perturbation into pre-logits, which are outputs of the penultimate layer, so as to maximize expected rewards with respect to the mean of the perturbation. We demonstrate that the optimal mean is obtained by importance sampling with sampled rewards. AISP outperforms best-of-n sampling in terms of rewards over the number of used samples and achieves higher rewards than other reward-based test-time alignment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26217v1">Hybrid LLM and Higher-Order Quantum Approximate Optimization for CSA Collateral Management</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      We address finance-native collateral optimization under ISDA Credit Support Annexes (CSAs), where integer lots, Schedule A haircuts, RA/MTA gating, and issuer/currency/class caps create rugged, legally bounded search spaces. We introduce a certifiable hybrid pipeline purpose-built for this domain: (i) an evidence-gated LLM that extracts CSA terms to a normalized JSON (abstain-by-default, span-cited); (ii) a quantum-inspired explorer that interleaves simulated annealing with micro higher order QAOA (HO-QAOA) on binding sub-QUBOs (subset size n <= 16, order k <= 4) to coordinate multi-asset moves across caps and RA-induced discreteness; (iii) a weighted risk-aware objective (Movement, CVaR, funding-priced overshoot) with an explicit coverage window U <= Reff+B; and (iv) CP-SAT as single arbiter to certify feasibility and gaps, including a U-cap pre-check that reports the minimal feasible buffer B*. Encoding caps/rounding as higher-order terms lets HO-QAOA target the domain couplings that defeat local swaps. On government bond datasets and multi-CSA inputs, the hybrid improves a strong classical baseline (BL-3) by 9.1%, 9.6%, and 10.7% across representative harnesses, delivering better cost-movement-tail frontiers under governance settings. We release governance grade artifacts-span citations, valuation matrix audit, weight provenance, QUBO manifests, and CP-SAT traces-to make results auditable and reproducible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26213v1">OmniLayout: Enabling Coarse-to-Fine Learning with LLMs for Universal Document Layout Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 TL;DR: With OmniLayout-1M dataset and LLM-based coarse-to-fine learning, we enable universal and diverse document layout generation
    </div>
    <details class="paper-abstract">
      Document AI has advanced rapidly and is attracting increasing attention. Yet, while most efforts have focused on document layout analysis (DLA), its generative counterpart, document layout generation, remains underexplored. A major obstacle lies in the scarcity of diverse layouts: academic papers with Manhattan-style structures dominate existing studies, while open-world genres such as newspapers and magazines remain severely underrepresented. To address this gap, we curate OmniLayout-1M, the first million-scale dataset of diverse document layouts, covering six common document types and comprising contemporary layouts collected from multiple sources. Moreover, since existing methods struggle in complex domains and often fail to arrange long sequences coherently, we introduce OmniLayout-LLM, a 0.5B model with designed two-stage Coarse-to-Fine learning paradigm: 1) learning universal layout principles from OmniLayout-1M with coarse category definitions, and 2) transferring the knowledge to a specific domain with fine-grained annotations. Extensive experiments demonstrate that our approach achieves strong performance on multiple domains in M$^{6}$Doc dataset, substantially surpassing both existing layout generation experts and several latest general-purpose LLMs. Our code, models, and dataset will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14970v4">Self-Evolving Curriculum for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective for fine-tuning large language models (LLMs), significantly enhancing their reasoning abilities in domains such as mathematics and code generation. A crucial factor influencing RL fine-tuning success is the training curriculum: the order in which training problems are presented. While random curricula serve as common baselines, they remain suboptimal; manually designed curricula often rely heavily on heuristics, and online filtering methods can be computationally prohibitive. To address these limitations, we propose Self-Evolving Curriculum (SEC), an automatic curriculum learning method that learns a curriculum policy concurrently with the RL fine-tuning process. Our approach formulates curriculum selection as a non-stationary Multi-Armed Bandit problem, treating each problem category (e.g., difficulty level or problem type) as an individual arm. We leverage the absolute advantage from policy gradient methods as a proxy measure for immediate learning gain. At each training step, the curriculum policy selects categories to maximize this reward signal and is updated using the TD(0) method. Across three distinct reasoning domains: planning, inductive reasoning, and mathematics, our experiments demonstrate that SEC significantly improves models' reasoning capabilities, enabling better generalization to harder, out-of-distribution test problems. Additionally, our approach achieves better skill balance when fine-tuning simultaneously on multiple reasoning domains. These findings highlight SEC as a promising strategy for RL fine-tuning of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16648v2">FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted in the Findings of EMNLP, 2025
    </div>
    <details class="paper-abstract">
      The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11094v2">The Scales of Justitia: A Comprehensive Survey on Safety Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 20 pages, preprint
    </div>
    <details class="paper-abstract">
      With the rapid advancement of artificial intelligence, Large Language Models (LLMs) have shown remarkable capabilities in Natural Language Processing (NLP), including content generation, human-computer interaction, machine translation, and code generation. However, their widespread deployment has also raised significant safety concerns. In particular, LLM-generated content can exhibit unsafe behaviors such as toxicity, bias, or misinformation, especially in adversarial contexts, which has attracted increasing attention from both academia and industry. Although numerous studies have attempted to evaluate these risks, a comprehensive and systematic survey on safety evaluation of LLMs is still lacking. This work aims to fill this gap by presenting a structured overview of recent advances in safety evaluation of LLMs. Specifically, we propose a four-dimensional taxonomy: (i) Why to evaluate, which explores the background of safety evaluation of LLMs, how they differ from general LLMs evaluation, and the significance of such evaluation; (ii) What to evaluate, which examines and categorizes existing safety evaluation tasks based on key capabilities, including dimensions such as toxicity, robustness, ethics, bias and fairness, truthfulness, and related aspects; (iii) Where to evaluate, which summarizes the evaluation metrics, datasets and benchmarks currently used in safety evaluations; (iv) How to evaluate, which reviews existing mainstream evaluation methods based on the roles of the evaluators and some evaluation frameworks that integrate the entire evaluation pipeline. Finally, we identify the challenges in safety evaluation of LLMs and propose promising research directions to promote further advancement in this field. We emphasize the necessity of prioritizing safety evaluation to ensure the reliable and responsible deployment of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26163v1">Exploring Dissatisfaction in Bus Route Reduction through LLM-Calibrated Agent-Based Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 17 pages, 8 figures, 4 tables
    </div>
    <details class="paper-abstract">
      As emerging mobility modes continue to expand, many cities face declining bus ridership, increasing fiscal pressure to sustain underutilized routes, and growing inefficiencies in resource allocation. This study employs an agent-based modelling (ABM) approach calibrated through a large language model (LLM) using few-shot learning to examine how progressive bus route cutbacks affect passenger dissatisfaction across demographic groups and overall network resilience. Using IC-card data from Beijing's Huairou District, the LLM-calibrated ABM estimated passenger sensitivity parameters related to travel time, waiting, transfers, and crowding. Results show that the structural configuration of the bus network exerts a stronger influence on system stability than capacity or operational factors. The elimination of high-connectivity routes led to an exponential rise in total dissatisfaction, particularly among passengers with disabilities and older adults. The evolution of dissatisfaction exhibited three distinct phases - stable, transitional, and critical. Through the analysis of each stage, this study found that the continuous bus route reduction scenario exhibits three-stage thresholds. Once these thresholds are crossed, even a small reduction in routes may lead to a significant loss of passenger flow. Research highlights the nonlinear response of user sentiment to service reductions and underscore the importance of maintaining structural critical routes and providing stable services to vulnerable groups for equitable and resilient transport planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26143v1">Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) can elicit strong reasoning in large language models (LLMs), yet most open efforts focus on math and code. We propose Reasoning Curriculum, a simple two-stage curriculum that first elicits reasoning skills in pretraining-aligned domains such as math, then adapts and refines these skills across other domains via joint RL. Stage 1 performs a brief cold start and then math-only RL with verifiable rewards to develop reasoning skills. Stage 2 runs joint RL on mixed-domain data to transfer and consolidate these skills. The curriculum is minimal and backbone-agnostic, requiring no specialized reward models beyond standard verifiability checks. Evaluated on Qwen3-4B and Llama-3.1-8B over a multi-domain suite, reasoning curriculum yields consistent gains. Ablations and a cognitive-skill analysis indicate that both stages are necessary and that math-first elicitation increases cognitive behaviors important for solving complex problems. Reasoning Curriculum provides a compact, easy-to-adopt recipe for general reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18477v2">LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 This paper has been accepted by the 16th IEEE International Conference on Cloud Computing Technology and Science (CloudCom 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great promise in automating data analytics tasks by interpreting natural language queries and generating multi-operation execution plans. However, existing LLM-agent-based analytics frameworks operate under the assumption of centralized data access, offering little to no privacy protection. In contrast, federated analytics (FA) enables privacy-preserving computation across distributed data sources, but lacks support for natural language input and requires structured, machine-readable queries. In this work, we present LAFA, the first system that integrates LLM-agent-based data analytics with FA. LAFA introduces a hierarchical multi-agent architecture that accepts natural language queries and transforms them into optimized, executable FA workflows. A coarse-grained planner first decomposes complex queries into sub-queries, while a fine-grained planner maps each subquery into a Directed Acyclic Graph of FA operations using prior structural knowledge. To improve execution efficiency, an optimizer agent rewrites and merges multiple DAGs, eliminating redundant operations and minimizing computational and communicational overhead. Our experiments demonstrate that LAFA consistently outperforms baseline prompting strategies by achieving higher execution plan success rates and reducing resource-intensive FA operations by a substantial margin. This work establishes a practical foundation for privacy-preserving, LLM-driven analytics that supports natural language input in the FA setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23595v3">Multi-Agent Evolve: LLM Self-Improve through Co-evolution</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 29 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26130v1">Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Pre-print prepared for journal submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced code generation at the function level, yet their ability to produce correct class-level implementations in authentic software projects remains poorly understood. This work introduces a novel benchmark derived from open-source repositories, comprising real-world classes divided into seen and unseen partitions to evaluate generalization under practical conditions. The evaluation examines multiple LLMs under varied input specifications, retrieval-augmented configurations, and documentation completeness levels. Results reveal a stark performance disparity: LLMs achieve 84% to 89% correctness on established synthetic benchmarks but only 25% to 34% on real-world class tasks, with negligible differences between familiar and novel codebases. Comprehensive docstrings yield modest gains of 1% to 3% in functional accuracy, though statistical significance is rare. Retrieval-augmented generation proves most effective with partial documentation, improving correctness by 4% to 7% by supplying concrete implementation patterns absent from specifications. Error profiling identifies AttributeError, TypeError, and AssertionError as dominant failure modes (84% of cases), with synthetic tests overemphasizing assertion issues and real-world scenarios highlighting type and attribute mismatches. Retrieval augmentation reduces logical flaws but can introduce dependency conflicts. The benchmark and analysis expose critical limitations in current LLM capabilities for class-level engineering, offering actionable insights for enhancing context modelling, documentation strategies, and retrieval integration in production code assistance tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26122v1">Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      While Test-Time Scaling (TTS) has proven effective in improving the reasoning ability of large language models (LLMs), low diversity in model outputs often becomes a bottleneck; this is partly caused by the common "one problem, one solution" (1P1S) training practice, which provides a single canonical answer and can push models toward a narrow set of reasoning paths. To address this, we propose a "one problem, multiple solutions" (1PNS) training paradigm that exposes the model to a variety of valid reasoning trajectories and thus increases inference diversity. A core challenge for 1PNS is reliably measuring semantic differences between multi-step chains of thought, so we introduce Reasoning Path Divergence (RPD), a step-level metric that aligns and scores Long Chain-of-Thought solutions to capture differences in intermediate reasoning. Using RPD, we curate maximally diverse solution sets per problem and fine-tune Qwen3-4B-Base. Experiments show that RPD-selected training yields more varied outputs and higher pass@k, with an average +2.80% gain in pass@16 over a strong 1P1S baseline and a +4.99% gain on AIME24, demonstrating that 1PNS further amplifies the effectiveness of TTS. Our code is available at https://github.com/fengjujf/Reasoning-Path-Divergence .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06736v3">Are LLMs Rigorous Logical Reasoners? Empowering Natural Language Proof Generation by Stepwise Decoding with Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 15 pages, 2 figures, 11 tables. Accepted by AACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Logical reasoning is a pivotal component in the field of artificial intelligence. Proof planning, particularly in contexts requiring the validation of explanation accuracy, continues to present challenges. The recent advancement of large language models (LLMs) has led to significant progress in natural language proof planning, evolving from one-stage generators to more complex three-stage systems that include additional searchers or verifiers. While these assisted methods improve the quality of generated results, they also introduce increased search efforts and computational costs. Furthermore, the generative process itself remains underexplored. In this study, we propose a stepwise decoding approach augmented by contrastive learning to address two common errors encountered during the LLM generator's decoding process. We fine-tune the language model using both vanilla and enhanced hard negatives to mitigate these decoding errors. Empirical results demonstrate the effectiveness of our strategy. Additionally, our further analysis reveals that even larger LLMs still struggle to generate rigorous logical chains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11695v2">When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Although Large Language Model (LLM)-based agents are increasingly used in financial trading, it remains unclear whether they can reason and adapt in live markets, as most studies test models instead of agents, cover limited periods and assets, and rely on unverified data. To address these gaps, we introduce Agent Market Arena (AMA), the first lifelong, real-time benchmark for evaluating LLM-based trading agents across multiple markets. AMA integrates verified trading data, expert-checked news, and diverse agent architectures within a unified trading framework, enabling fair and continuous comparison under real conditions. It implements four agents, including InvestorAgent as a single-agent baseline, TradeAgent and HedgeFundAgent with different risk styles, and DeepFundAgent with memory-based reasoning, and evaluates them across GPT-4o, GPT-4.1, Claude-3.5-haiku, Claude-sonnet-4, and Gemini-2.0-flash. Live experiments on both cryptocurrency and stock markets demonstrate that agent frameworks display markedly distinct behavioral patterns, spanning from aggressive risk-taking to conservative decision-making, whereas model backbones contribute less to outcome variation. AMA thus establishes a foundation for rigorous, reproducible, and continuously evolving evaluation of financial reasoning and trading intelligence in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14808v4">HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 NeurIPS 2025. 21 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have facilitated a wide range of applications with distinct service-level objectives (SLOs), from latency-sensitive online tasks like interactive chatbots to throughput-oriented offline workloads like data synthesis. The existing deployment model, which dedicates machines to each workload, simplifies SLO management but often leads to poor resource utilization. This paper introduces HyGen, an interference-aware LLM serving system that enables efficient co-location of online and offline workloads while preserving SLOs. HyGen incorporates two key innovations: (1) performance control mechanisms, including a latency predictor to estimate batch execution time and an SLO-aware profiler to quantify latency interference, and (2) SLO-aware offline scheduling policies that maximize serving throughput and prevent starvation. Our evaluation on production workloads shows that HyGen achieves up to 3.9-5.8x throughput gains over online and hybrid serving baselines, while ensuring latency SLOs. The code of HyGen is publicly available at https://github.com/UIUC-MLSys/HyGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03710v3">Improving LLM Safety Alignment with Dual-Objective Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26037v1">SIRAJ: Diverse and Efficient Red-Teaming for LLM Agents via Distilled Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      The ability of LLM agents to plan and invoke tools exposes them to new safety risks, making a comprehensive red-teaming system crucial for discovering vulnerabilities and ensuring their safe deployment. We present SIRAJ: a generic red-teaming framework for arbitrary black-box LLM agents. We employ a dynamic two-step process that starts with an agent definition and generates diverse seed test cases that cover various risk outcomes, tool-use trajectories, and risk sources. Then, it iteratively constructs and refines model-based adversarial attacks based on the execution trajectories of former attempts. To optimize the red-teaming cost, we present a model distillation approach that leverages structured forms of a teacher model's reasoning to train smaller models that are equally effective. Across diverse evaluation agent settings, our seed test case generation approach yields 2 -- 2.5x boost to the coverage of risk outcomes and tool-calling trajectories. Our distilled 8B red-teamer model improves attack success rate by 100%, surpassing the 671B Deepseek-R1 model. Our ablations and analyses validate the effectiveness of the iterative framework, structured reasoning, and the generalization of our red-teamer models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27016v1">Semantically-Aware LLM Agent to Enhance Privacy in Conversational AI Services</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted to IEEE Big Data 2025
    </div>
    <details class="paper-abstract">
      With the increasing use of conversational AI systems, there is growing concern over privacy leaks, especially when users share sensitive personal data in interactions with Large Language Models (LLMs). Conversations shared with these models may contain Personally Identifiable Information (PII), which, if exposed, could lead to security breaches or identity theft. To address this challenge, we present the Local Optimizations for Pseudonymization with Semantic Integrity Directed Entity Detection (LOPSIDED) framework, a semantically-aware privacy agent designed to safeguard sensitive PII data when using remote LLMs. Unlike prior work that often degrade response quality, our approach dynamically replaces sensitive PII entities in user prompts with semantically consistent pseudonyms, preserving the contextual integrity of conversations. Once the model generates its response, the pseudonyms are automatically depseudonymized, ensuring the user receives an accurate, privacy-preserving output. We evaluate our approach using real-world conversations sourced from ShareGPT, which we further augment and annotate to assess whether named entities are contextually relevant to the model's response. Our results show that LOPSIDED reduces semantic utility errors by a factor of 5 compared to baseline techniques, all while enhancing privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00413v2">Accelerating Diffusion LLMs via Adaptive Parallel Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The generation speed of LLMs are bottlenecked by autoregressive decoding, where tokens are predicted sequentially one by one. Alternatively, diffusion large language models (dLLMs) theoretically allow for parallel token generation, but in practice struggle to achieve the speed of autoregressive models without significantly sacrificing quality. We therefore introduce adaptive parallel decoding (APD), a novel method that dynamically adjusts the number of tokens sampled in parallel. We achieve this by defining a multiplicative mixture between the dLLM marginal probabilities and the joint probability of sequences under a small auxiliary autoregressive model. This inverts the standard setup of speculative decoding, where the goal is to sample from a large autoregressive verifier by drafting from a smaller model. We further optimize APD by enabling KV caching and limiting the size of the masked input. Altogether, our method puts forward three tunable parameters to flexibly tradeoff throughput and quality. We show that APD provides markedly higher throughput with minimal quality degradations on downstream benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26995v1">LLMs are Overconfident: Evaluating Confidence Interval Calibration with FermiEval</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at numerical estimation but struggle to correctly quantify uncertainty. We study how well LLMs construct confidence intervals around their own answers and find that they are systematically overconfident. To evaluate this behavior, we introduce FermiEval, a benchmark of Fermi-style estimation questions with a rigorous scoring rule for confidence interval coverage and sharpness. Across several modern models, nominal 99\% intervals cover the true answer only 65\% of the time on average. With a conformal prediction based approach that adjusts the intervals, we obtain accurate 99\% observed coverage, and the Winkler interval score decreases by 54\%. We also propose direct log-probability elicitation and quantile adjustment methods, which further reduce overconfidence at high confidence levels. Finally, we develop a perception-tunnel theory explaining why LLMs exhibit overconfidence: when reasoning under uncertainty, they act as if sampling from a truncated region of their inferred distribution, neglecting its tails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19669v2">DiffAdapt: Difficulty-Adaptive Reasoning for Token-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Recent reasoning Large Language Models (LLMs) demonstrate remarkable problem-solving abilities but often generate long thinking traces whose utility is unclear. Our work aims to improve their efficiency, enabling them to reach high performance without overthinking. First, we analyze the entropy of token probabilities in reasoning traces. Across three models, we observe a consistent U-shaped entropy pattern: high entropy on easy problems despite high accuracy, low entropy on problems with medium difficulty, and high entropy on hard problems reflecting uncertainty. Specifically, we notice 22--25\% entropy reduction from easy to medium difficulty regions, suggesting an {overthinking} phenomenon on easy instances. Building on these insights, we introduce \textbf{DiffAdapt}, a lightweight framework that selects Easy/Normal/Hard inference strategies per question based on their difficulty and reasoning trace entropy. Each inference strategy consists of a fixed prompt, temperature and maximum token length. In contrast to existing efficiency optimization methods, our approach does not fine-tune base LLM but a small probe that classifies LLM's final hidden state, allowing inexpensive adaptation. We comprehensively evaluate our method on five models and eight benchmarks. Our method achieves comparable or improved accuracy while reducing token usage by up to 22.4\%, establishing a practical path toward compute-efficient reasoning.
    </details>
</div>
