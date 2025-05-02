# llm - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17331v1">Exploring Context-aware and LLM-driven Locomotion for Immersive Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Locomotion plays a crucial role in shaping the user experience within virtual reality environments. In particular, hands-free locomotion offers a valuable alternative by supporting accessibility and freeing users from reliance on handheld controllers. To this end, traditional speech-based methods often depend on rigid command sets, limiting the naturalness and flexibility of interaction. In this study, we propose a novel locomotion technique powered by large language models (LLMs), which allows users to navigate virtual environments using natural language with contextual awareness. We evaluate three locomotion methods: controller-based teleportation, voice-based steering, and our language model-driven approach. Our evaluation measures include eye-tracking data analysis, including explainable machine learning through SHAP analysis as well as standardized questionnaires for usability, presence, cybersickness, and cognitive load to examine user attention and engagement. Our findings indicate that the LLM-driven locomotion possesses comparable usability, presence, and cybersickness scores to established methods like teleportation, demonstrating its novel potential as a comfortable, natural language-based, hands-free alternative. In addition, it enhances user attention within the virtual environment, suggesting greater engagement. Complementary to these findings, SHAP analysis revealed that fixation, saccade, and pupil-related features vary across techniques, indicating distinct patterns of visual attention and cognitive processing. Overall, we state that our method can facilitate hands-free locomotion in virtual spaces, especially in supporting accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13471v2">From Large to Super-Tiny: End-to-End Optimization for Cost-Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have significantly advanced artificial intelligence by optimizing traditional Natural Language Processing (NLP) pipelines, improving performance and generalization. This has spurred their integration into various systems. Many NLP systems, including ours, employ a "one-stage" pipeline directly incorporating LLMs. While effective, this approach incurs substantial costs and latency due to the need for large model parameters to achieve satisfactory outcomes. This paper introduces a three-stage cost-efficient end-to-end LLM deployment pipeline-including prototyping, knowledge transfer, and model compression-to tackle the cost-performance dilemma in LLM-based frameworks. Our approach yields a super tiny model optimized for cost and performance in online systems, simplifying the system architecture. Initially, by transforming complex tasks into a function call-based LLM-driven pipeline, an optimal performance prototype system is constructed to produce high-quality data as a teacher model. The second stage combines techniques like rejection fine-tuning, reinforcement learning, and knowledge distillation to transfer knowledge to a smaller 0.5B student model, delivering effective performance at minimal cost. The final stage applies quantization and pruning to extremely compress models to 0.4B, achieving ultra-low latency and cost. The framework's modular design and cross-domain capabilities suggest potential applicability in other NLP areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17226v1">FLAG: Formal and LLM-assisted SVA Generation for Formal Specifications of On-Chip Communication Protocols</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 9 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Formal specifications of on-chip communication protocols are crucial for system-on-chip (SoC) design and verification. However, manually constructing these formal specifications from informal documents remains a tedious and error-prone task. Although recent efforts have used Large Language Models (LLMs) to generate SystemVerilog Assertion (SVA) properties from design documents for Register-Transfer Level (RTL) design verification, in our experience these approaches have not shown promise in generating SVA properties for communication protocols. Since protocol specification documents are unstructured and ambiguous in nature, LLMs often fail to extract the necessary information and end up generating irrelevant or even incorrect properties. We propose FLAG, a two-stage framework to help construct formal protocol specifications from informal documents. In the first stage, a predefined template set is used to generate candidate SVA properties. To avoid missing necessary properties, we develop a grammar-based approach to generate comprehensive template sets that capture critical signal behaviors for various communication protocols. In the second stage, we utilize unambiguous timing diagrams in conjunction with textual descriptions from the specification documents to filter out incorrect properties. A formal approach is first implemented to check the candidate properties and filter out those inconsistent with the timing diagrams. An LLM is then consulted to further remove incorrect properties with respect to the textual description, obtaining the final property set. Experiments on various open-source communication protocols demonstrate the effectiveness of FLAG in generating SVA properties from informal documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17200v1">A RAG-Based Multi-Agent LLM System for Natural Hazard Resilience and Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are a transformational capability at the frontier of artificial intelligence and machine learning that can support decision-makers in addressing pressing societal challenges such as extreme natural hazard events. As generalized models, LLMs often struggle to provide context-specific information, particularly in areas requiring specialized knowledge. In this work we propose a retrieval-augmented generation (RAG)-based multi-agent LLM system to support analysis and decision-making in the context of natural hazards and extreme weather events. As a proof of concept, we present WildfireGPT, a specialized system focused on wildfire hazards. The architecture employs a user-centered, multi-agent design to deliver tailored risk insights across diverse stakeholder groups. By integrating natural hazard and extreme weather projection data, observational datasets, and scientific literature through an RAG framework, the system ensures both the accuracy and contextual relevance of the information it provides. Evaluation across ten expert-led case studies demonstrates that WildfireGPT significantly outperforms existing LLM-based solutions for decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13192v2">CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Accepted by KDD 2024;
    </div>
    <details class="paper-abstract">
      Recently, Large Language Model (LLM)-empowered recommender systems (RecSys) have brought significant advances in personalized user experience and have attracted considerable attention. Despite the impressive progress, the research question regarding the safety vulnerability of LLM-empowered RecSys still remains largely under-investigated. Given the security and privacy concerns, it is more practical to focus on attacking the black-box RecSys, where attackers can only observe the system's inputs and outputs. However, traditional attack approaches employing reinforcement learning (RL) agents are not effective for attacking LLM-empowered RecSys due to the limited capabilities in processing complex textual inputs, planning, and reasoning. On the other hand, LLMs provide unprecedented opportunities to serve as attack agents to attack RecSys because of their impressive capability in simulating human-like decision-making processes. Therefore, in this paper, we propose a novel attack framework called CheatAgent by harnessing the human-like capabilities of LLMs, where an LLM-based agent is developed to attack LLM-Empowered RecSys. Specifically, our method first identifies the insertion position for maximum impact with minimal input modification. After that, the LLM agent is designed to generate adversarial perturbations to insert at target positions. To further improve the quality of generated perturbations, we utilize the prompt tuning technique to improve attacking strategies via feedback from the victim RecSys iteratively. Extensive experiments across three real-world datasets demonstrate the effectiveness of our proposed attacking method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06425v2">Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Accepted to The ACM Web Conference (WWW) 2025 Short Paper Track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly utilized in domains such as finance, healthcare, and interpersonal relationships to provide advice tailored to user traits and contexts. However, this personalization often relies on sensitive data, raising critical privacy concerns and necessitating data minimization. To address these challenges, we propose a framework that integrates zero-knowledge proof (ZKP) technology, specifically zkVM, with LLM-based chatbots. This integration enables privacy-preserving data sharing by verifying user traits without disclosing sensitive information. Our research introduces both an architecture and a prompting strategy for this approach. Through empirical evaluation, we clarify the current constraints and performance limitations of both zkVM and the proposed prompting strategy, thereby demonstrating their practical feasibility in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17967v1">LLM Agent Swarm for Hypothesis-Driven Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 15 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Drug discovery remains a formidable challenge: more than 90 percent of candidate molecules fail in clinical evaluation, and development costs often exceed one billion dollars per approved therapy. Disparate data streams, from genomics and transcriptomics to chemical libraries and clinical records, hinder coherent mechanistic insight and slow progress. Meanwhile, large language models excel at reasoning and tool integration but lack the modular specialization and iterative memory required for regulated, hypothesis-driven workflows. We introduce PharmaSwarm, a unified multi-agent framework that orchestrates specialized LLM "agents" to propose, validate, and refine hypotheses for novel drug targets and lead compounds. Each agent accesses dedicated functionality--automated genomic and expression analysis; a curated biomedical knowledge graph; pathway enrichment and network simulation; interpretable binding affinity prediction--while a central Evaluator LLM continuously ranks proposals by biological plausibility, novelty, in silico efficacy, and safety. A shared memory layer captures validated insights and fine-tunes underlying submodels over time, yielding a self-improving system. Deployable on low-code platforms or Kubernetes-based microservices, PharmaSwarm supports literature-driven discovery, omics-guided target identification, and market-informed repurposing. We also describe a rigorous four-tier validation pipeline spanning retrospective benchmarking, independent computational assays, experimental testing, and expert user studies to ensure transparency, reproducibility, and real-world impact. By acting as an AI copilot, PharmaSwarm can accelerate translational research and deliver high-confidence hypotheses more efficiently than traditional pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06621v5">Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 ICLR 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong reasoning and memorization capabilities via pretraining on massive textual corpora. However, this poses risk of privacy and copyright violations, highlighting the need for efficient machine unlearning methods that remove sensitive data without retraining from scratch. While Gradient Ascent (GA) is commonly used to unlearn by reducing the likelihood of generating unwanted content, it leads to unstable optimization and catastrophic forgetting of retrained knowledge. We find that combining GA with low-rank adaptation results in poor trade-offs between computational cost and generative performance. To address these challenges, we propose Low-rank Knowledge Unlearning (LoKU), a novel framework that enables robust and efficient unlearning for LLMs. First, we introduce Inverted Hinge Loss, which suppresses unwanted tokens while maintaining fluency by boosting the probability of the next most likely token. Second, we develop a data-adaptive initialization for LoRA adapters via low-rank approximation weighted with relative Fisher information, thereby focusing updates on parameters critical for removing targeted knowledge. Experiments on the Training Data Extraction Challenge dataset using GPT-Neo models as well as on the TOFU benchmark with Phi-1.5B and Llama2-7B models demonstrate that our approach effectively removes sensitive information while maintaining reasoning and generative capabilities with minimal impact. Our implementation can be found in https://github.com/csm9493/efficient-llm-unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17950v1">Collaborating Action by Action: A Multi-agent LLM Framework for Embodied Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 9 pages of main paper with 6 main figures, overall 28 pages
    </div>
    <details class="paper-abstract">
      Collaboration is ubiquitous and essential in day-to-day life -- from exchanging ideas, to delegating tasks, to generating plans together. This work studies how LLMs can adaptively collaborate to perform complex embodied reasoning tasks. To this end we introduce MINDcraft, an easily extensible platform built to enable LLM agents to control characters in the open-world game of Minecraft; and MineCollab, a benchmark to test the different dimensions of embodied and collaborative reasoning. An experimental study finds that the primary bottleneck in collaborating effectively for current state-of-the-art agents is efficient natural language communication, with agent performance dropping as much as 15% when they are required to communicate detailed task completion plans. We conclude that existing LLM agents are ill-optimized for multi-agent collaboration, especially in embodied scenarios, and highlight the need to employ methods beyond in-context and imitation learning. Our website can be found here: https://mindcraft-minecollab.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17934v1">Toward a Human-Centered Evaluation Framework for Trustworthy LLM-Powered GUI Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has revolutionized Graphical User Interface (GUI) automation through LLM-powered GUI agents, yet their ability to process sensitive data with limited human oversight raises significant privacy and security risks. This position paper identifies three key risks of GUI agents and examines how they differ from traditional GUI automation and general autonomous agents. Despite these risks, existing evaluations focus primarily on performance, leaving privacy and security assessments largely unexplored. We review current evaluation metrics for both GUI and general LLM agents and outline five key challenges in integrating human evaluators for GUI agent assessments. To address these gaps, we advocate for a human-centered evaluation framework that incorporates risk assessments, enhances user awareness through in-context consent, and embeds privacy and security considerations into GUI agent design and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08040v2">Can Reasoning LLMs Enhance Clinical Document Classification?</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Clinical document classification is essential for converting unstructured medical texts into standardised ICD-10 diagnoses, yet it faces challenges due to complex medical language, privacy constraints, and limited annotated datasets. Large Language Models (LLMs) offer promising improvements in accuracy and efficiency for this task. This study evaluates the performance and consistency of eight LLMs; four reasoning (Qwen QWQ, Deepseek Reasoner, GPT o3 Mini, Gemini 2.0 Flash Thinking) and four non-reasoning (Llama 3.3, GPT 4o Mini, Gemini 2.0 Flash, Deepseek Chat); in classifying clinical discharge summaries using the MIMIC-IV dataset. Using cTAKES to structure clinical narratives, models were assessed across three experimental runs, with majority voting determining final predictions. Results showed that reasoning models outperformed non-reasoning models in accuracy (71% vs 68%) and F1 score (67% vs 60%), with Gemini 2.0 Flash Thinking achieving the highest accuracy (75%) and F1 score (76%). However, non-reasoning models demonstrated greater stability (91% vs 84% consistency). Performance varied across ICD-10 codes, with reasoning models excelling in complex cases but struggling with abstract categories. Findings indicate a trade-off between accuracy and consistency, suggesting that a hybrid approach could optimise clinical coding. Future research should explore multi-label classification, domain-specific fine-tuning, and ensemble methods to enhance model reliability in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16266v2">TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on edge platforms is challenged by their high computational and memory demands. Although recent low-bit quantization methods (e.g., BitNet, DeepSeek) compress weights to as little as 1.58 bits with minimal accuracy loss, edge deployment is still constrained by limited on-chip resources, power budgets, and the often-neglected latency of the prefill phase. We present TeLLMe, the first ternary LLM accelerator for low-power FPGAs (e.g., AMD KV260) that fully supports both prefill and autoregressive decoding using 1.58-bit weights and 8-bit activations. Our contributions include: (1) a table-lookup matrix engine for ternary matmul that merges grouped activations with online precomputation to minimize resource use; (2) a fused, bandwidth-efficient attention module featuring a reversed reordering scheme to accelerate prefill; and (3) a tightly integrated normalization and quantization--dequantization unit optimized for ultra-low-bit inference. Under a 7W power budget, TeLLMe delivers up to 9 tokens/s throughput over 1,024-token contexts and prefill latencies of 0.55--1.15 s for 64--128 token prompts, marking a significant energy-efficiency advance and establishing a new edge FPGA benchmark for generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17833v1">The Role of Open-Source LLMs in Shaping the Future of GeoAI</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming geospatial artificial intelligence (GeoAI), offering new capabilities in data processing, spatial analysis, and decision support. This paper examines the open-source paradigm's pivotal role in this transformation. While proprietary LLMs offer accessibility, they often limit the customization, interoperability, and transparency vital for specialized geospatial tasks. Conversely, open-source alternatives significantly advance Geographic Information Science (GIScience) by fostering greater adaptability, reproducibility, and community-driven innovation. Open frameworks empower researchers to tailor solutions, integrate cutting-edge methodologies (e.g., reinforcement learning, advanced spatial indexing), and align with FAIR principles. However, the growing reliance on any LLM necessitates careful consideration of security vulnerabilities, ethical risks, and robust governance for AI-generated geospatial outputs. Ongoing debates on accessibility, regulation, and misuse underscore the critical need for responsible AI development strategies. This paper argues that GIScience advances best not through a single model type, but by cultivating a diverse, interoperable ecosystem combining open-source foundations for innovation, bespoke geospatial models, and interdisciplinary collaboration. By critically evaluating the opportunities and challenges of open-source LLMs within the broader GeoAI landscape, this work contributes to a nuanced discourse on leveraging AI to effectively advance spatial research, policy, and decision-making in an equitable, sustainable, and scientifically rigorous manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18603v1">Toward Personalizing Quantum Computing Education: An Evolutionary LLM-Powered Approach</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Quantum computing education faces significant challenges due to its complexity and the limitations of current tools; this paper introduces a novel Intelligent Teaching Assistant for quantum computing education and details its evolutionary design process. The system combines a knowledge-graph-augmented architecture with two specialized Large Language Model (LLM) agents: a Teaching Agent for dynamic interaction, and a Lesson Planning Agent for lesson plan generation. The system is designed to adapt to individual student needs, with interactions meticulously tracked and stored in a knowledge graph. This graph represents student actions, learning resources, and relationships, aiming to enable reasoning about effective learning pathways. We describe the implementation of the system, highlighting the challenges encountered and the solutions implemented, including introducing a dual-agent architecture where tasks are separated, all coordinated through a central knowledge graph that maintains system awareness, and a user-facing tag system intended to mitigate LLM hallucination and improve user control. Preliminary results illustrate the system's potential to capture rich interaction data, dynamically adapt lesson plans based on student feedback via a tag system in simulation, and facilitate context-aware tutoring through the integrated knowledge graph, though systematic evaluation is required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18598v1">BadMoE: Backdooring Mixture-of-Experts LLMs via Optimizing Routing Triggers and Infecting Dormant Experts</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) have emerged as a powerful architecture for large language models (LLMs), enabling efficient scaling of model capacity while maintaining manageable computational costs. The key advantage lies in their ability to route different tokens to different ``expert'' networks within the model, enabling specialization and efficient handling of diverse input. However, the vulnerabilities of MoE-based LLMs still have barely been studied, and the potential for backdoor attacks in this context remains largely unexplored. This paper presents the first backdoor attack against MoE-based LLMs where the attackers poison ``dormant experts'' (i.e., underutilized experts) and activate them by optimizing routing triggers, thereby gaining control over the model's output. We first rigorously prove the existence of a few ``dominating experts'' in MoE models, whose outputs can determine the overall MoE's output. We also show that dormant experts can serve as dominating experts to manipulate model predictions. Accordingly, our attack, namely \textsc{BadMoE}, exploits the unique architecture of MoE models by 1) identifying dormant experts unrelated to the target task, 2) constructing a routing-aware loss to optimize the activation triggers of these experts, and 3) promoting dormant experts to dominating roles via poisoned training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20073v1">RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) as interactive agents presents unique challenges including long-horizon decision making and interacting with stochastic environment feedback. While reinforcement learning (RL) has enabled progress in static tasks, multi-turn agent RL training remains underexplored. We propose StarPO (State-Thinking-Actions-Reward Policy Optimization), a general framework for trajectory-level agent RL, and introduce RAGEN, a modular system for training and evaluating LLM agents. Our study on three stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of Echo Trap where reward variance cliffs and gradient spikes; we address this with StarPO-S, a stabilized variant with trajectory filtering, critic incorporation, and decoupled clipping. Second, we find the shaping of RL rollouts would benefit from diverse initial states, medium interaction granularity and more frequent sampling. Third, we show that without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge through multi-turn RL and they may show shallow strategies or hallucinated thoughts. Code and environments are available at https://github.com/RAGEN-AI/RAGEN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20068v1">Tempo: Application-aware LLM Serving with Mixed SLO Requirements</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into diverse applications, ranging from interactive chatbots and cloud AIOps to intelligent agents, has introduced a wide spectrum of Service Level Objectives (SLOs) for responsiveness. These workloads include latency-sensitive requests focused on per-token latency in streaming chat, throughput-intensive requests that require rapid full responses to invoke tools, and collective requests with dynamic dependencies arising from self-reflection or agent-based reasoning. This workload diversity, amplified by unpredictable request information such as response lengths and runtime dependencies, makes existing schedulers inadequate even within their design envelopes. In this paper, we define service gain as the useful service delivered by completing requests. We observe that as SLO directly reflects the actual performance needs of requests, completing a request much faster than its SLO (e.g., deadline) yields limited additional service gain. Based on this insight, we introduce Tempo, the first systematic SLO-aware scheduler designed to maximize service gain across diverse LLM workloads. Tempo allocates just enough serving bandwidth to meet each SLO, maximizing residual capacity for others best-effort workloads. Instead of assuming request information or none at all, it adopts a hybrid scheduling strategy: using quantile-based response upper bounds and dependency-graph matching for conservative initial estimates, prioritizing requests by service gain density, and refining decisions online as generation progresses. Our evaluation across diverse workloads, including chat, reasoning, and agentic pipelines, shows that Tempo improves end-to-end service gain by up to 8.3$\times$ and achieves up to 10.3$\times$ SLO goodput compared to state-of-the-art designs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17768v1">The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Sparse attention offers a promising strategy to extend long-context capabilities in Transformer LLMs, yet its viability, its efficiency-accuracy trade-offs, and systematic scaling studies remain unexplored. To address this gap, we perform a careful comparison of training-free sparse attention methods at varying model scales, sequence lengths, and sparsity levels on a diverse collection of long-sequence tasks-including novel ones that rely on natural language while remaining controllable and easy to evaluate. Based on our experiments, we report a series of key findings: 1) an isoFLOPS analysis reveals that for very long sequences, larger and highly sparse models are preferable to smaller and dense ones. 2) The level of sparsity attainable while statistically guaranteeing accuracy preservation is higher during decoding than prefilling, and correlates with model size in the former. 3) There is no clear strategy that performs best across tasks and phases, with different units of sparsification or budget adaptivity needed for different scenarios. Even moderate sparsity levels often result in significant performance degradation on at least one task, highlighting that sparse attention is not a universal solution. 4) We introduce and validate novel scaling laws specifically tailored for sparse attention, providing evidence that our findings are likely to hold true beyond our range of experiments. Through these insights, we demonstrate that sparse attention is a key tool to enhance the capabilities of Transformer LLMs for processing longer sequences, but requires careful evaluation of trade-offs for performance-sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17723v1">Towards Robust LLMs: an Adversarial Robustness Measurement Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has revolutionized artificial intelligence, yet these models remain vulnerable to adversarial perturbations, undermining their reliability in high-stakes applications. While adversarial robustness in vision-based neural networks has been extensively studied, LLM robustness remains under-explored. We adapt the Robustness Measurement and Assessment (RoMA) framework to quantify LLM resilience against adversarial inputs without requiring access to model parameters. By comparing RoMA's estimates to those of formal verification methods, we demonstrate its accuracy with minimal error margins while maintaining computational efficiency. Our empirical evaluation reveals that robustness varies significantly not only between different models but also across categories within the same task and between various types of perturbations. This non-uniformity underscores the need for task-specific robustness evaluations, enabling practitioners to compare and select models based on application-specific robustness requirements. Our work provides a systematic methodology to assess LLM robustness, advancing the development of more reliable language models for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17685v1">Ensemble Bayesian Inference: Leveraging Small Language Models to Achieve LLM-level Accuracy in Profile Matching Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      This study explores the potential of small language model(SLM) ensembles to achieve accuracy comparable to proprietary large language models (LLMs). We propose Ensemble Bayesian Inference (EBI), a novel approach that applies Bayesian estimation to combine judgments from multiple SLMs, allowing them to exceed the performance limitations of individual models. Our experiments on diverse tasks(aptitude assessments and consumer profile analysis in both Japanese and English) demonstrate EBI's effectiveness. Notably, we analyze cases where incorporating models with negative Lift values into ensembles improves overall performance, and we examine the method's efficacy across different languages. These findings suggest new possibilities for constructing high-performance AI systems with limited computational resources and for effectively utilizing models with individually lower performance. Building on existing research on LLM performance evaluation, ensemble methods, and open-source LLM utilization, we discuss the novelty and significance of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16871v2">Exploring How LLMs Capture and Represent Domain-Specific Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      We study whether Large Language Models (LLMs) inherently capture domain-specific nuances in natural language. Our experiments probe the domain sensitivity of LLMs by examining their ability to distinguish queries from different domains using hidden states generated during the prefill phase. We reveal latent domain-related trajectories that indicate the model's internal recognition of query domains. We also study the robustness of these domain representations to variations in prompt styles and sources. Our approach leverages these representations for model selection, mapping the LLM that best matches the domain trace of the input query (i.e., the model with the highest performance on similar traces). Our findings show that LLMs can differentiate queries for related domains, and that the fine-tuned model is not always the most accurate. Unlike previous work, our interpretations apply to both closed and open-ended generative tasks
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11242v4">Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Published at ICLR 2025 (Oral)
    </div>
    <details class="paper-abstract">
      LLMs are an integral component of retrieval-augmented generation (RAG) systems. While many studies focus on evaluating the overall quality of end-to-end RAG systems, there is a gap in understanding the appropriateness of LLMs for the RAG task. To address this, we introduce Trust-Score, a holistic metric that evaluates the trustworthiness of LLMs within the RAG framework. Our results show that various prompting methods, such as in-context learning, fail to effectively adapt LLMs to the RAG task as measured by Trust-Score. Consequently, we propose Trust-Align, a method to align LLMs for improved Trust-Score performance. 26 out of 27 models aligned using Trust-Align substantially outperform competitive baselines on ASQA, QAMPARI, and ELI5. Specifically, in LLaMA-3-8b, Trust-Align outperforms FRONT on ASQA (up 12.56), QAMPARI (up 36.04), and ELI5 (up 17.69). Trust-Align also significantly enhances models' ability to correctly refuse and provide quality citations. We also demonstrate the effectiveness of Trust-Align across different open-weight models, including the LLaMA series (1b to 8b), Qwen-2.5 series (0.5b to 7b), and Phi3.5 (3.8b). We release our code at https://github.com/declare-lab/trust-align.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12533v3">To Help or Not to Help: LLM-based Attentive Support for Human-Robot Group Interactions</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024
    </div>
    <details class="paper-abstract">
      How can a robot provide unobtrusive physical support within a group of humans? We present Attentive Support, a novel interaction concept for robots to support a group of humans. It combines scene perception, dialogue acquisition, situation understanding, and behavior generation with the common-sense reasoning capabilities of Large Language Models (LLMs). In addition to following user instructions, Attentive Support is capable of deciding when and how to support the humans, and when to remain silent to not disturb the group. With a diverse set of scenarios, we show and evaluate the robot's attentive behavior, which supports and helps the humans when required, while not disturbing if no help is needed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17584v1">L3: DIMM-PIM Integrated Architecture and Coordination for Scalable Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 16 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly require processing long text sequences, but GPU memory limitations force difficult trade-offs between memory capacity and bandwidth. While HBM-based acceleration offers high bandwidth, its capacity remains constrained. Offloading data to host-side DIMMs improves capacity but introduces costly data swapping overhead. We identify that the critical memory bottleneck lies in the decoding phase of multi-head attention (MHA) exclusively, which demands substantial capacity for storing KV caches and high bandwidth for attention computation. Our key insight reveals this operation uniquely aligns with modern DIMM-based processing-in-memory (PIM) architectures, which offers scalability of both capacity and bandwidth. Based on this observation and insight, we propose L3, a hardware-software co-designed system integrating DIMM-PIM and GPU devices. L3 introduces three innovations: First, hardware redesigns resolve data layout mismatches and computational element mismatches in DIMM-PIM, enhancing LLM inference utilization. Second, communication optimization enables hiding the data transfer overhead with the computation. Third, an adaptive scheduler coordinates GPU-DIMM-PIM operations to maximize parallelism between devices. Evaluations using real-world traces show L3 achieves up to 6.1$\times$ speedup over state-of-the-art HBM-PIM solutions while significantly improving batch sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07347v2">Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      As demand for Large Language Models (LLMs) and AI agents rapidly grows, optimizing systems for efficient LLM inference becomes critical. While significant efforts have focused on system-level engineering, little is explored from a mathematical modeling and queuing perspective. In this paper, we aim to develop the queuing fundamentals for large language model (LLM) inference, bridging the gap between the queueing theory and LLM system communities. In particular, we study the throughput aspect in LLM inference systems. We prove that a large class of 'work-conserving' scheduling algorithms can achieve maximum throughput for individual inference LLM engine, highlighting 'work-conserving' as a key design principle in practice. In a network of LLM agents, work-conserving scheduling alone is insufficient, particularly when facing specific workload structures and multi-class workflows that require more sophisticated scheduling strategies. Evaluations of real-world systems show that Orca and Sarathi-serve are throughput-optimal, reassuring practitioners, while FasterTransformer and vanilla vLLM are not maximally stable and should be used with caution. Our results highlight the substantial benefits that the queueing community can offer in improving LLM inference systems and call for more interdisciplinary development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17565v1">DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have recently achieved remarkable performance on various complex reasoning benchmarks, the academic community still lacks an in-depth understanding of base model training processes and data quality. To address this, we construct a large-scale, difficulty-graded reasoning dataset containing approximately 3.34 million unique queries of varying difficulty levels and about 40 million distilled responses generated by multiple models over several passes. Leveraging pass rate and Coefficient of Variation (CV), we precisely select the most valuable training data to enhance reasoning capability. Notably, we observe a training pattern shift, indicating that reasoning-focused training based on base models requires higher learning rates for effective training. Using this carefully selected data, we significantly improve the reasoning capabilities of the base model, achieving a pass rate of 79.2\% on the AIME2024 mathematical reasoning benchmark. This result surpasses most current distilled models and closely approaches state-of-the-art performance. We provide detailed descriptions of our data processing, difficulty assessment, and training methodology, and have publicly released all datasets and methods to promote rapid progress in open-source long-reasoning LLMs. The dataset is available at: https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17550v1">HalluLens: LLM Hallucination Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate responses that deviate from user input or training data, a phenomenon known as "hallucination." These hallucinations undermine user trust and hinder the adoption of generative AI systems. Addressing hallucinations is essential for the advancement of LLMs. This paper introduces a comprehensive hallucination benchmark, incorporating both new extrinsic and existing intrinsic evaluation tasks, built upon clear taxonomy of hallucination. A major challenge in benchmarking hallucinations is the lack of a unified framework due to inconsistent definitions and categorizations. We disentangle LLM hallucination from "factuality," proposing a clear taxonomy that distinguishes between extrinsic and intrinsic hallucinations, to promote consistency and facilitate research. Extrinsic hallucinations, where the generated content is not consistent with the training data, are increasingly important as LLMs evolve. Our benchmark includes dynamic test set generation to mitigate data leakage and ensure robustness against such leakage. We also analyze existing benchmarks, highlighting their limitations and saturation. The work aims to: (1) establish a clear taxonomy of hallucinations, (2) introduce new extrinsic hallucination tasks, with data that can be dynamically regenerated to prevent saturation by leakage, (3) provide a comprehensive analysis of existing benchmarks, distinguishing them from factuality evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02868v2">Enhancing LLMs with Smart Preprocessing for EHR Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency in natural language processing; however, their application in sensitive domains such as healthcare, especially in processing Electronic Health Records (EHRs), is constrained by limited computational resources and privacy concerns. This paper introduces a compact LLM framework optimized for local deployment in environments with stringent privacy requirements and restricted access to high-performance GPUs. Our approach leverages simple yet powerful preprocessing techniques, including regular expressions (regex) and Retrieval-Augmented Generation (RAG), to extract and highlight critical information from clinical notes. By pre-filtering long, unstructured text, we enhance the performance of smaller LLMs on EHR-related tasks. Our framework is evaluated using zero-shot and few-shot learning paradigms on both private and publicly available datasets (MIMIC-IV), with additional comparisons against fine-tuned LLMs on MIMIC-IV. Experimental results demonstrate that our preprocessing strategy significantly supercharges the performance of smaller LLMs, making them well-suited for privacy-sensitive and resource-constrained applications. This study offers valuable insights into optimizing LLM performance for local, secure, and efficient healthcare applications. It provides practical guidance for real-world deployment for LLMs while tackling challenges related to privacy, computational feasibility, and clinical applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17497v1">Combining GCN Structural Learning with LLM Chemical Knowledge for or Enhanced Virtual Screening</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Virtual screening plays a critical role in modern drug discovery by enabling the identification of promising candidate molecules for experimental validation. Traditional machine learning methods such as support vector machines (SVM) and XGBoost rely on predefined molecular representations, often leading to information loss and potential bias. In contrast, deep learning approaches-particularly Graph Convolutional Networks (GCNs)-offer a more expressive and unbiased alternative by operating directly on molecular graphs. Meanwhile, Large Language Models (LLMs) have recently demonstrated state-of-the-art performance in drug design, thanks to their capacity to capture complex chemical patterns from large-scale data via attention mechanisms. In this paper, we propose a hybrid architecture that integrates GCNs with LLM-derived embeddings to combine localized structural learning with global chemical knowledge. The LLM embeddings can be precomputed and stored in a molecular feature library, removing the need to rerun the LLM during training or inference and thus maintaining computational efficiency. We found that concatenating the LLM embeddings after each GCN layer-rather than only at the final layer-significantly improves performance, enabling deeper integration of global context throughout the network. The resulting model achieves superior results, with an F1-score of (88.8%), outperforming standalone GCN (87.9%), XGBoost (85.5%), and SVM (85.4%) baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17445v1">Creating Targeted, Interpretable Topic Models with LLM-Generated Text Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Presented at IC2S2 2024 in Philadelphia, USA
    </div>
    <details class="paper-abstract">
      Unsupervised machine learning techniques, such as topic modeling and clustering, are often used to identify latent patterns in unstructured text data in fields such as political science and sociology. These methods overcome common concerns about reproducibility and costliness involved in the labor-intensive process of human qualitative analysis. However, two major limitations of topic models are their interpretability and their practicality for answering targeted, domain-specific social science research questions. In this work, we investigate opportunities for using LLM-generated text augmentation to improve the usefulness of topic modeling output. We use a political science case study to evaluate our results in a domain-specific application, and find that topic modeling using GPT-4 augmentations creates highly interpretable categories that can be used to investigate domain-specific research questions with minimal human guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01503v4">A Highly Scalable LLM Clusters with Optical Interconnect</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      We propose \emph{LumosCore} to build high-bandwidth and large-scale data center networks for LLM jobs. By replacing the core-layer electrical packet switches by optical circuit switches, \emph{LumosCore} could achieves $2\times$ increase in bandwidth or $8\times$ increase in network size. We offer the detailed design of \emph{LumosCore} at both deployment stage and running stage. At deployment stage, we propose Interleaved Wiring, which is compatible with all possible logical topologies. At running stage, we design polynomial-time algorithms for GPU placement, logical topology generating and OCS reconfiguration to minimize network contention and reduce impact to scheduled jobs. We evaluate \emph{LumosCore} using both testbed experiments and large-scale simulation. Compared to traditional hybrid optical/electrical architectures, \emph{LumosCore} increases the end-to-end training throughput by up to 39.5\% on a 128-node testbed. Compared to the state-of-art Clos architectures, \emph{LumosCore} reduces the average job completion time by up to 34.1\% in a 16k simulation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12022v3">ITERTL: An Iterative Framework for Fine-tuning LLMs for RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated excellent performance, inspiring researchers to explore their use in automating register transfer level (RTL) code generation and improving hardware design efficiency. However, the existing approaches to fine-tune LLMs for RTL generation typically are conducted on fixed datasets, which do not fully stimulate the capability of LLMs and require large amounts of reference data, which are costly to acquire. To mitigate these issues, we innovatively introduce an iterative training paradigm named ITERTL. During each iteration, samples are drawn from the model trained in the previous cycle. Then these new samples are employed for training in current loop. Furthermore, we introduce a plug-and-play data filtering strategy, thereby encouraging the model to generate high-quality, self-contained code. Our model outperforms GPT4 and state-of-the-art (SOTA) open-source models, achieving remarkable 53.8% pass@1 rate on VerilogEval-human benchmark. Under similar conditions of data quantity and quality, our approach significantly outperforms the baseline. Extensive experiments validate the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19572v5">ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Accepted at Conference of the North American Chapter of the Association for Computational Linguistics, Student Research Workshop 2025 (NAACL SRW 2025)
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) systems using large language models (LLMs) often generate inaccurate responses due to the retrieval of irrelevant or loosely related information. Existing methods, which operate at the document level, fail to effectively filter out such content. We propose LLM-driven chunk filtering, ChunkRAG, a framework that enhances RAG systems by evaluating and filtering retrieved information at the chunk level. Our approach employs semantic chunking to divide documents into coherent sections and utilizes LLM-based relevance scoring to assess each chunk's alignment with the user's query. By filtering out less pertinent chunks before the generation phase, we significantly reduce hallucinations and improve factual accuracy. Experiments show that our method outperforms existing RAG models, achieving higher accuracy on tasks requiring precise information retrieval. This advancement enhances the reliability of RAG systems, making them particularly beneficial for applications like fact-checking and multi-hop reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10157v2">SocioVerse: A World Model for Social Simulation Powered by LLM Agents and A Pool of 10 Million Real-World Users</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      Social simulation is transforming traditional social science research by modeling human behavior through interactions between virtual individuals and their environments. With recent advances in large language models (LLMs), this approach has shown growing potential in capturing individual differences and predicting group behaviors. However, existing methods face alignment challenges related to the environment, target users, interaction mechanisms, and behavioral patterns. To this end, we introduce SocioVerse, an LLM-agent-driven world model for social simulation. Our framework features four powerful alignment components and a user pool of 10 million real individuals. To validate its effectiveness, we conducted large-scale simulation experiments across three distinct domains: politics, news, and economics. Results demonstrate that SocioVerse can reflect large-scale population dynamics while ensuring diversity, credibility, and representativeness through standardized procedures and minimal manual adjustments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16429v1">Give LLMs a Security Course: Securing Retrieval-Augmented Code Generation via Knowledge Injection</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Code Generation (RACG) leverages external knowledge to enhance Large Language Models (LLMs) in code synthesis, improving the functional correctness of the generated code. However, existing RACG systems largely overlook security, leading to substantial risks. Especially, the poisoning of malicious code into knowledge bases can mislead LLMs, resulting in the generation of insecure outputs, which poses a critical threat in modern software development. To address this, we propose a security-hardening framework for RACG systems, CodeGuarder, that shifts the paradigm from retrieving only functional code examples to incorporating both functional code and security knowledge. Our framework constructs a security knowledge base from real-world vulnerability databases, including secure code samples and root cause annotations. For each code generation query, a retriever decomposes the query into fine-grained sub-tasks and fetches relevant security knowledge. To prioritize critical security guidance, we introduce a re-ranking and filtering mechanism by leveraging the LLMs' susceptibility to different vulnerability types. This filtered security knowledge is seamlessly integrated into the generation prompt. Our evaluation shows CodeGuarder significantly improves code security rates across various LLMs, achieving average improvements of 20.12\% in standard RACG, and 31.53\% and 21.91\% under two distinct poisoning scenarios without compromising functional correctness. Furthermore, CodeGuarder demonstrates strong generalization, enhancing security even when the targeted language's security knowledge is lacking. This work presents CodeGuarder as a pivotal advancement towards building secure and trustworthy RACG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20770v4">Large Language Model Sentinel: LLM Agent for Adversarial Purification</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15610v2">A LoRA-Based Approach to Fine-Tuning LLMs for Educational Guidance in Resource-Constrained Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 18 pages, 6 figures (3 graphs + 3 flowchart/architecture diagrams), submitted as a preprint for review consideration in AI for Education or Machine Learning applications in low-resource settings. Includes detailed experiments with LoRA and quantization methods for efficient LLM fine-tuning
    </div>
    <details class="paper-abstract">
      The current study describes a cost-effective method for adapting large language models (LLMs) for academic advising with study-abroad contexts in mind and for application in low-resource methods for acculturation. With the Mistral-7B-Instruct model applied with a Low-Rank Adaptation (LoRA) method and a 4-bit quantization method, the model underwent training in two distinct stages related to this study's purpose to enhance domain specificity while maintaining computational efficiency. In Phase 1, the model was conditioned with a synthetic dataset via the Gemini Pro API, and in Phase 2, it was trained with manually curated datasets from the StudyAbroadGPT project to achieve enhanced, contextualized responses. Technical innovations entailed memory-efficient quantization, parameter-efficient adaptation, and continuous training analytics via Weights & Biases. After training, this study demonstrated a reduction in training loss by 52.7%, 92% accuracy in domain-specific recommendations, achieved 95% markdown-based formatting support, and a median run-rate of 100 samples per second on off-the-shelf GPU equipment. These findings support the effective application of instruction-tuned LLMs within educational advisers, especially in low-resource institutional scenarios. Limitations included decreased generalizability and the application of a synthetically generated dataset, but this framework is scalable for adding new multilingual-augmented and real-time academic advising processes. Future directions may include plans for the integration of retrieval-augmented generation, applying dynamic quantization routines, and connecting to real-time academic databases to increase adaptability and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13217v2">Sustainability via LLM Right-sizing</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 17 pages, 2 Figures, 6 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly embedded in organizational workflows. This has raised concerns over their energy consumption, financial costs, and data sovereignty. While performance benchmarks often celebrate cutting-edge models, real-world deployment decisions require a broader perspective: when is a smaller, locally deployable model "good enough"? This study offers an empirical answer by evaluating eleven proprietary and open-weight LLMs across ten everyday occupational tasks, including summarizing texts, generating schedules, and drafting emails and proposals. Using a dual-LLM-based evaluation framework, we automated task execution and standardized evaluation across ten criteria related to output quality, factual accuracy, and ethical responsibility. Results show that GPT-4o delivers consistently superior performance but at a significantly higher cost and environmental footprint. Notably, smaller models like Gemma-3 and Phi-4 achieved strong and reliable results on most tasks, suggesting their viability in contexts requiring cost-efficiency, local deployment, or privacy. A cluster analysis revealed three model groups -- premium all-rounders, competent generalists, and limited but safe performers -- highlighting trade-offs between quality, control, and sustainability. Significantly, task type influenced model effectiveness: conceptual tasks challenged most models, while aggregation and transformation tasks yielded better performances. We argue for a shift from performance-maximizing benchmarks to task- and context-aware sufficiency assessments that better reflect organizational priorities. Our approach contributes a scalable method to evaluate AI models through a sustainability lens and offers actionable guidance for responsible LLM deployment in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06845v4">7B Fully Open Source Moxin-LLM -- From Pretraining to GRPO-based Reinforcement Learning Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Although open-source LLMs present unprecedented opportunities for innovation and research, the commercialization of LLMs has raised concerns about transparency, reproducibility, and safety. Many open-source LLMs fail to meet fundamental transparency requirements by withholding essential components like training code and data, which may hinder further innovations on LLMs. To mitigate this issue, we introduce Moxin 7B, a fully open-source LLM developed, adhering to principles of open science, open source, open data, and open access. We release the pre-training code and configurations, training and fine-tuning datasets, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. After pre-training and obtaining the base model, we finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our Instruct model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization (GRPO), an efficient and effective reinforcement learning algorithm following DeepSeek R1, to finetune our model, leading to the Moxin Reasoning model. Experiments show that our models achieve superior performance in various evaluations such as zero-shot evaluation, few-shot evaluation, and CoT evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00348v2">Attention Tracker: Detecting Prompt Injection Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Project page: https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05401v2">Post-hoc Study of Climate Microtargeting on Social Media Ads with LLMs: Thematic Insights and Fairness Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Climate change communication on social media increasingly employs microtargeting strategies to effectively reach and influence specific demographic groups. This study presents a post-hoc analysis of microtargeting practices within climate campaigns by leveraging large language models (LLMs) to examine Facebook advertisements. Our analysis focuses on two key aspects: demographic targeting and fairness. We evaluate the ability of LLMs to accurately predict the intended demographic targets, such as gender and age group, achieving an overall accuracy of 88.55%. Furthermore, we instruct the LLMs to generate explanations for their classifications, providing transparent reasoning behind each decision. These explanations reveal the specific thematic elements used to engage different demographic segments, highlighting distinct strategies tailored to various audiences. Our findings show that young adults are primarily targeted through messages emphasizing activism and environmental consciousness, while women are engaged through themes related to caregiving roles and social advocacy. In addition to evaluating the effectiveness of LLMs in detecting microtargeted messaging, we conduct a comprehensive fairness analysis to identify potential biases in model predictions. Our findings indicate that while LLMs perform well overall, certain biases exist, particularly in the classification of senior citizens and male audiences. By showcasing the efficacy of LLMs in dissecting and explaining targeted communication strategies and by highlighting fairness concerns, this study provides a valuable framework for future research aimed at enhancing transparency, accountability, and inclusivity in social media-driven climate campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17130v1">Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have transformed the way we access information. These models are often tuned to refuse to comply with requests that are considered harmful and to produce responses that better align with the preferences of those who control the models. To understand how this "censorship" works. We use representation engineering techniques to study open-weights safety-tuned models. We present a method for finding a refusal--compliance vector that detects and controls the level of censorship in model outputs. We also analyze recent reasoning LLMs, distilled from DeepSeek-R1, and uncover an additional dimension of censorship through "thought suppression". We show a similar approach can be used to find a vector that suppresses the model's reasoning process, allowing us to remove censorship by applying the negative multiples of this vector
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17087v1">Leveraging LLMs as Meta-Judges: A Multi-Agent Framework for Evaluating LLM Judgments</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 12 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being widely applied across various fields, but as tasks become more complex, evaluating their responses is increasingly challenging. Compared to human evaluators, the use of LLMs to support performance evaluation offers a more efficient alternative. However, most studies focus mainly on aligning LLMs' judgments with human preferences, overlooking the existence of biases and mistakes in human judgment. Furthermore, how to select suitable LLM judgments given multiple potential LLM responses remains underexplored. To address these two aforementioned issues, we propose a three-stage meta-judge selection pipeline: 1) developing a comprehensive rubric with GPT-4 and human experts, 2) using three advanced LLM agents to score judgments, and 3) applying a threshold to filter out low-scoring judgments. Compared to methods using a single LLM as both judge and meta-judge, our pipeline introduces multi-agent collaboration and a more comprehensive rubric. Experimental results on the JudgeBench dataset show about 15.55\% improvement compared to raw judgments and about 8.37\% improvement over the single-agent baseline. Our work demonstrates the potential of LLMs as meta-judges and lays the foundation for future research on constructing preference datasets for LLM-as-a-judge reinforcement learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17083v1">How Individual Traits and Language Styles Shape Preferences In Open-ended User-LLM Interaction: A Preliminary Study</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Accepted at GenAICHI 2025 @ ACM CHI 2025
    </div>
    <details class="paper-abstract">
      What makes an interaction with the LLM more preferable for the user? While it is intuitive to assume that information accuracy in the LLM's responses would be one of the influential variables, recent studies have found that inaccurate LLM's responses could still be preferable when they are perceived to be more authoritative, certain, well-articulated, or simply verbose. These variables interestingly fall under the broader category of language style, implying that the style in the LLM's responses might meaningfully influence users' preferences. This hypothesized dynamic could have double-edged consequences: enhancing the overall user experience while simultaneously increasing their susceptibility to risks such as LLM's misinformation or hallucinations. In this short paper, we present our preliminary studies in exploring this subject. Through a series of exploratory and experimental user studies, we found that LLM's language style does indeed influence user's preferences, but how and which language styles influence the preference varied across different user populations, and more interestingly, moderated by the user's very own individual traits. As a preliminary work, the findings in our studies should be interpreted with caution, particularly given the limitations in our samples, which still need wider demographic diversity and larger sample sizes. Our future directions will first aim to address these limitations, which would enable a more comprehensive joint effect analysis between the language style, individual traits, and preferences, and further investigate the potential causal relationship between and beyond these variables.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17075v1">Agree to Disagree? A Meta-Evaluation of LLM Misgendering</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Numerous methods have been proposed to measure LLM misgendering, including probability-based evaluations (e.g., automatically with templatic sentences) and generation-based evaluations (e.g., with automatic heuristics or human validation). However, it has gone unexamined whether these evaluation methods have convergent validity, that is, whether their results align. Therefore, we conduct a systematic meta-evaluation of these methods across three existing datasets for LLM misgendering. We propose a method to transform each dataset to enable parallel probability- and generation-based evaluation. Then, by automatically evaluating a suite of 6 models from 3 families, we find that these methods can disagree with each other at the instance, dataset, and model levels, conflicting on 20.2% of evaluation instances. Finally, with a human evaluation of 2400 LLM generations, we show that misgendering behaviour is complex and goes far beyond pronouns, which automatic evaluations are not currently designed to capture, suggesting essential disagreement with human evaluations. Based on our findings, we provide recommendations for future evaluations of LLM misgendering. Our results are also more widely relevant, as they call into question broader methodological conventions in LLM evaluation, which often assume that different evaluation methods agree.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17070v1">Robo-Troj: Attacking LLM-based Task Planners</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17025v1">Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      The number of pretrained Large Language Models (LLMs) is increasing steadily, though the majority are designed predominantly for the English language. While state-of-the-art LLMs can handle other languages, due to language contamination or some degree of multilingual pretraining data, they are not optimized for non-English languages, leading to inefficient encoding (high token "fertility") and slower inference speed. In this work, we thoroughly compare a variety of vocabulary adaptation techniques for optimizing English LLMs for the Italian language, and put forward Semantic Alignment Vocabulary Adaptation (SAVA), a novel method that leverages neural mapping for vocabulary substitution. SAVA achieves competitive performance across multiple downstream tasks, enhancing grounded alignment strategies. We adapt two LLMs: Mistral-7b-v0.1, reducing token fertility by 25\%, and Llama-3.1-8B, optimizing the vocabulary and reducing the number of parameters by 1 billion. We show that, following the adaptation of the vocabulary, these models can recover their performance with a relatively limited stage of continual training on the target language. Finally, we test the capabilities of the adapted models on various multi-choice and generative tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17018v1">LLM impact on BLV programming</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Submitted to ASSETS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly becoming integral to a wide range of tools, tasks, and problem-solving processes, especially in software development. Originally designed for natural language processing tasks such as text generation, LLMs are increasingly being used to assist both professionals and students in writing code. This growing reliance on LLM-based tools is reshaping programming workflows and task execution. In this study, we explore the impact of these technologies on blind and low-vision (BLV) developers. Our review of existing literature indicates that while LLMs help mitigate some of the challenges faced by BLV programmers, they also introduce new forms of inaccessibility. We conducted an evaluation of five popular LLM-powered integrated development environments (IDEs), assessing their performance across a comprehensive set of programming tasks. Our findings highlight several unsupported scenarios, instances of incorrect model output, and notable limitations in interaction support for specific tasks. Through observing BLV developers as they engaged in coding activities, we uncovered key interaction barriers that go beyond model accuracy or code generation quality. This paper outlines the challenges and corresponding opportunities for improving accessibility in the context of generative AI-assisted programming. Addressing these issues can meaningfully enhance the programming experience for BLV developers. As the generative AI revolution continues to unfold, it must also address the unique burdens faced by this community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15364v2">KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 8 pages, 14 figures
    </div>
    <details class="paper-abstract">
      In this work, we demonstrate that distinctive keys during LLM inference tend to have high attention scores. We explore this phenomenon and propose KeyDiff, a training-free KV cache eviction method based on key similarity. This method facilitates the deployment of LLM-based application requiring long input prompts in resource-constrained environments with limited memory and compute budgets. Unlike other KV cache eviction methods, KeyDiff can process arbitrarily long prompts within strict resource constraints and efficiently generate responses. We demonstrate that KeyDiff computes the optimal solution to a KV cache selection problem that maximizes key diversity, providing a theoretical understanding of KeyDiff. Notably,KeyDiff does not rely on attention scores, allowing the use of optimized attention mechanisms like FlashAttention. We demonstrate the effectiveness of KeyDiff across diverse tasks and models, illustrating a performance gap of less than 0.04\% with 8K cache budget ($\sim$ 23\% KV cache reduction) from the non-evicting baseline on the LongBench benchmark for Llama 3.1-8B and Llama 3.2-3B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15903v2">Impact of Noise on LLM-Models Performance in Abstraction and Reasoning Corpus (ARC) Tasks with Model Temperature Considerations</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 60 pages, 25 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have generated growing interest in their structured reasoning capabilities, particularly in tasks involving abstraction and pattern recognition. The Abstraction and Reasoning Corpus (ARC) benchmark plays a crucial role in evaluating these capabilities by testing how well AI models generalize to novel problems. While GPT-4o demonstrates strong performance by solving all ARC tasks under zero-noise conditions, other models like DeepSeek R1 and LLaMA 3.2 fail to solve any, suggesting limitations in their ability to reason beyond simple pattern matching. To explore this gap, we systematically evaluate these models across different noise levels and temperature settings. Our results reveal that the introduction of noise consistently impairs model performance, regardless of architecture. This decline highlights a shared vulnerability: current LLMs, despite showing signs of abstract reasoning, remain highly sensitive to input perturbations. Such fragility raises concerns about their real-world applicability, where noise and uncertainty are common. By comparing how different model architectures respond to these challenges, we offer insights into the structural weaknesses of modern LLMs in reasoning tasks. This work underscores the need for developing more robust and adaptable AI systems capable of handling the ambiguity and variability inherent in real-world scenarios. Our findings aim to guide future research toward enhancing model generalization, robustness, and alignment with human-like cognitive flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17824v1">EduBot -- Can LLMs Solve Personalized Learning and Programming Assignments?</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Published at AAAI 2025 AI4EDU Workshop
    </div>
    <details class="paper-abstract">
      The prevalence of Large Language Models (LLMs) is revolutionizing the process of writing code. General and code LLMs have shown impressive performance in generating standalone functions and code-completion tasks with one-shot queries. However, the ability to solve comprehensive programming tasks with recursive requests and bug fixes remains questionable. In this paper, we propose EduBot, an intelligent automated assistant system that combines conceptual knowledge teaching, end-to-end code development, personalized programming through recursive prompt-driven methods, and debugging with limited human interventions powered by LLMs. We show that EduBot can solve complicated programming tasks consisting of sub-tasks with increasing difficulties ranging from conceptual to coding questions by recursive automatic prompt-driven systems without finetuning on LLMs themselves. To further evaluate EduBot's performance, we design and conduct a benchmark suite consisting of 20 scenarios in algorithms, machine learning, and real-world problems. The result shows that EduBot can complete most scenarios in less than 20 minutes. Based on the benchmark suites, we perform a comparative study to take different LLMs as the backbone and to verify EduBot's compatibility and robustness across LLMs with varying capabilities. We believe that EduBot is an exploratory approach to explore the potential of pre-trained LLMs in multi-step reasoning and code generation for solving personalized assignments with knowledge learning and code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18583v1">PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The autoregressive nature of large language models (LLMs) limits inference speed. Each forward pass generates only a single token and is often bottlenecked by memory bandwidth. Speculative decoding alleviates this issue using a draft-then-verify approach to accelerate token generation. However, the overhead introduced during the draft phase and the training cost of the draft model limit the efficiency and adaptability of speculative decoding. In this work, we introduce PARallel Draft (PARD), a novel speculative decoding method that enables low-cost adaptation of autoregressive draft models into parallel draft models. PARD enhances inference efficiency by predicting multiple future tokens in a single forward pass of the draft phase, and incorporates a conditional drop token method to accelerate training. Its target-independence property allows a single draft model to be applied to an entire family of different models, minimizing the adaptation cost. Our proposed conditional drop token method can improves draft model training efficiency by 3x. On our optimized inference framework, PARD accelerates LLaMA3.1-8B inference by 4.08x, achieving 311.5 tokens per second.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16921v1">IberBench: LLM Evaluation on Iberian Languages</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) remain difficult to evaluate comprehensively, particularly for languages other than English, where high-quality data is often limited. Existing benchmarks and leaderboards are predominantly English-centric, with only a few addressing other languages. These benchmarks fall short in several key areas: they overlook the diversity of language varieties, prioritize fundamental Natural Language Processing (NLP) capabilities over tasks of industrial relevance, and are static. With these aspects in mind, we present IberBench, a comprehensive and extensible benchmark designed to assess LLM performance on both fundamental and industry-relevant NLP tasks, in languages spoken across the Iberian Peninsula and Ibero-America. IberBench integrates 101 datasets from evaluation campaigns and recent benchmarks, covering 22 task categories such as sentiment and emotion analysis, toxicity detection, and summarization. The benchmark addresses key limitations in current evaluation practices, such as the lack of linguistic diversity and static evaluation setups by enabling continual updates and community-driven model and dataset submissions moderated by a committee of experts. We evaluate 23 LLMs ranging from 100 million to 14 billion parameters and provide empirical insights into their strengths and limitations. Our findings indicate that (i) LLMs perform worse on industry-relevant tasks than in fundamental ones, (ii) performance is on average lower for Galician and Basque, (iii) some tasks show results close to random, and (iv) in other tasks LLMs perform above random but below shared task systems. IberBench offers open-source implementations for the entire evaluation pipeline, including dataset normalization and hosting, incremental evaluation of LLMs, and a publicly accessible leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16918v1">OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Optimization plays a vital role in scientific research and practical applications, but formulating a concrete optimization problem described in natural language into a mathematical form and selecting a suitable solver to solve the problem requires substantial domain expertise. We introduce \textbf{OptimAI}, a framework for solving \underline{Optim}ization problems described in natural language by leveraging LLM-powered \underline{AI} agents, achieving superior performance over current state-of-the-art methods. Our framework is built upon four key roles: (1) a \emph{formulator} that translates natural language problem descriptions into precise mathematical formulations; (2) a \emph{planner} that constructs a high-level solution strategy prior to execution; and (3) a \emph{coder} and a \emph{code critic} capable of interacting with the environment and reflecting on outcomes to refine future actions. Ablation studies confirm that all roles are essential; removing the planner or code critic results in $5.8\times$ and $3.1\times$ drops in productivity, respectively. Furthermore, we introduce UCB-based debug scheduling to dynamically switch between alternative plans, yielding an additional $3.3\times$ productivity gain. Our design emphasizes multi-agent collaboration, allowing us to conveniently explore the synergistic effect of combining diverse models within a unified system. Our approach attains 88.1\% accuracy on the NLP4LP dataset and 71.2\% on the Optibench (non-linear w/o table) subset, reducing error rates by 58\% and 50\% respectively over prior best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16913v1">Tracing Thought: Using Chain-of-Thought Reasoning to Identify the LLM Behind AI-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 De-Factify 4: 4th Workshop on Multimodal Fact Checking and Hate Speech Detection, co-located with AAAI 2025. Pennsylvania
    </div>
    <details class="paper-abstract">
      In recent years, the detection of AI-generated text has become a critical area of research due to concerns about academic integrity, misinformation, and ethical AI deployment. This paper presents COT Fine-tuned, a novel framework for detecting AI-generated text and identifying the specific language model. responsible for generating the text. We propose a dual-task approach, where Task A involves classifying text as AI-generated or human-written, and Task B identifies the specific LLM behind the text. The key innovation of our method lies in the use of Chain-of-Thought reasoning, which enables the model to generate explanations for its predictions, enhancing transparency and interpretability. Our experiments demonstrate that COT Fine-tuned achieves high accuracy in both tasks, with strong performance in LLM identification and human-AI classification. We also show that the CoT reasoning process contributes significantly to the models effectiveness and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14985v2">aiXamine: Simplified LLM Safety and Security</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16871v1">Exploring How LLMs Capture and Represent Domain-Specific Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      We study whether Large Language Models (LLMs) inherently capture domain-specific nuances in natural language. Our experiments probe the domain sensitivity of LLMs by examining their ability to distinguish queries from different domains using hidden states generated during the prefill phase. We reveal latent domain-related trajectories that indicate the model's internal recognition of query domains. We also study the robustness of these domain representations to variations in prompt styles and sources. Our approach leverages these representations for model selection, mapping the LLM that best matches the domain trace of the input query (i.e., the model with the highest performance on similar traces). Our findings show that LLMs can differentiate queries for related domains, and that the fine-tuned model is not always the most accurate. Unlike previous work, our interpretations apply to both closed and open-ended generative tasks
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16046v2">Certified Mitigation of Worst-Case LLM Copyright Infringement</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      The exposure of large language models (LLMs) to copyrighted material during pre-training raises concerns about unintentional copyright infringement post deployment. This has driven the development of "copyright takedown" methods, post-training approaches aimed at preventing models from generating content substantially similar to copyrighted ones. While current mitigation approaches are somewhat effective for average-case risks, we demonstrate that they overlook worst-case copyright risks exhibits by the existence of long, verbatim quotes from copyrighted sources. We propose BloomScrub, a remarkably simple yet highly effective inference-time approach that provides certified copyright takedown. Our method repeatedly interleaves quote detection with rewriting techniques to transform potentially infringing segments. By leveraging efficient data sketches (Bloom filters), our approach enables scalable copyright screening even for large-scale real-world corpora. When quotes beyond a length threshold cannot be removed, the system can abstain from responding, offering certified risk reduction. Experimental results show that BloomScrub reduces infringement risk, preserves utility, and accommodates different levels of enforcement stringency with adaptive abstention. Our results suggest that lightweight, inference-time methods can be surprisingly effective for copyright prevention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00694v2">On Benchmarking Code LLMs for Android Malware Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 This paper has been accepted to the 34th ACM SIGSOFT ISSTA Companion (LLMSC Workshop 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in various code intelligence tasks. However, their effectiveness for Android malware analysis remains underexplored. Decompiled Android malware code presents unique challenges for analysis, due to the malicious logic being buried within a large number of functions and the frequent lack of meaningful function names. This paper presents CAMA, a benchmarking framework designed to systematically evaluate the effectiveness of Code LLMs in Android malware analysis. CAMA specifies structured model outputs to support key malware analysis tasks, including malicious function identification and malware purpose summarization. Built on these, it integrates three domain-specific evaluation metrics (consistency, fidelity, and semantic relevance), enabling rigorous stability and effectiveness assessment and cross-model comparison. We construct a benchmark dataset of 118 Android malware samples from 13 families collected in recent years, encompassing over 7.5 million distinct functions, and use CAMA to evaluate four popular open-source Code LLMs. Our experiments provide insights into how Code LLMs interpret decompiled code and quantify the sensitivity to function renaming, highlighting both their potential and current limitations in malware analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16833v1">LRASGen: LLM-based RESTful API Specification Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      REpresentation State Transfer (REST) is an architectural style for designing web applications that enable scalable, stateless communication between clients and servers via common HTTP techniques. Web APIs that employ the REST style are known as RESTful (or REST) APIs. When using or testing a RESTful API, developers may need to employ its specification, which is often defined by open-source standards such as the OpenAPI Specification (OAS). However, it can be very time-consuming and error-prone to write and update these specifications, which may negatively impact the use of RESTful APIs, especially when the software requirements change. Many tools and methods have been proposed to solve this problem, such as Respector and Swagger Core. OAS generation can be regarded as a common text-generation task that creates a formal description of API endpoints derived from the source code. A potential solution for this may involve using Large Language Models (LLMs), which have strong capabilities in both code understanding and text generation. Motivated by this, we propose a novel approach for generating the OASs of RESTful APIs using LLMs: LLM-based RESTful API-Specification Generation (LRASGen). To the best of our knowledge, this is the first use of LLMs and API source code to generate OASs for RESTful APIs. Compared with existing tools and methods, LRASGen can generate the OASs, even when the implementation is incomplete (with partial code, and/or missing annotations/comments, etc.). To evaluate the LRASGen performance, we conducted a series of empirical studies on 20 real-world RESTful APIs. The results show that two LLMs (GPT-4o mini and DeepSeek V3) can both support LARSGen to generate accurate specifications, and LRASGen-generated specifications cover an average of 48.85% more missed entities than the developer-provided specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16813v1">LLM-assisted Graph-RAG Information Extraction from IFC Data</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 2025 European Conference on Computing in Construction
    </div>
    <details class="paper-abstract">
      IFC data has become the general building information standard for collaborative work in the construction industry. However, IFC data can be very complicated because it allows for multiple ways to represent the same product information. In this research, we utilise the capabilities of LLMs to parse the IFC data with Graph Retrieval-Augmented Generation (Graph-RAG) technique to retrieve building object properties and their relations. We will show that, despite limitations due to the complex hierarchy of the IFC data, the Graph-RAG parsing enhances generative LLMs like GPT-4o with graph-based knowledge, enabling natural language query-response retrieval without the need for a complex pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05812v2">Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Ongoing work. First released on April 8, 2025. Updated the natural reasoning results on April 23, 2025
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated exceptional capabilities in challenging tasks such as mathematical reasoning, existing methods to enhance reasoning ability predominantly rely on supervised fine-tuning (SFT) followed by reinforcement learning (RL) on reasoning-specific data after pre-training. However, these approaches critically depend on external supervision--such as human-labelled reasoning traces, verified golden answers, or pre-trained reward models--which limits scalability and practical applicability. In this work, we propose Entropy Minimized Policy Optimization (EMPO), which makes an early attempt at fully unsupervised LLM reasoning incentivization. EMPO does not require any supervised information for incentivizing reasoning capabilities (i.e., neither verifiable reasoning traces, problems with golden answers, nor additional pre-trained reward models). By continuously minimizing the predictive entropy of LLMs on unlabeled user queries in a latent semantic space, EMPO enables purely self-supervised evolution of reasoning capabilities with strong flexibility and practicality. Our experiments demonstrate competitive performance of EMPO on both mathematical reasoning and free-form natural reasoning tasks. Specifically, without any supervised signals, \ours boosts the accuracy of Qwen2.5-Math-7B Base from 30.7\% to 48.1\% on mathematical benchmarks and improves the accuracy of Qwen2.5-7B Base from 32.1\% to 50.1\% on MMLU-Pro.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10982v4">Exploring the Role of Knowledge Graph-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15965v2">From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 26 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Memory is the process of encoding, storing, and retrieving information, allowing humans to retain experiences, knowledge, skills, and facts over time, and serving as the foundation for growth and effective interaction with the world. It plays a crucial role in shaping our identity, making decisions, learning from past experiences, building relationships, and adapting to changes. In the era of large language models (LLMs), memory refers to the ability of an AI system to retain, recall, and use information from past interactions to improve future responses and interactions. Although previous research and reviews have provided detailed descriptions of memory mechanisms, there is still a lack of a systematic review that summarizes and analyzes the relationship between the memory of LLM-driven AI systems and human memory, as well as how we can be inspired by human memory to construct more powerful memory systems. To achieve this, in this paper, we propose a comprehensive survey on the memory of LLM-driven AI systems. In particular, we first conduct a detailed analysis of the categories of human memory and relate them to the memory of AI systems. Second, we systematically organize existing memory-related work and propose a categorization method based on three dimensions (object, form, and time) and eight quadrants. Finally, we illustrate some open problems regarding the memory of current AI systems and outline possible future directions for memory in the era of large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00513v2">Leveraging LLMs for User Stories in AI Systems: UStAI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.11094v2">Reinforcement Learning With LLMs Interaction For Distributed Diffusion Model Services</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Distributed Artificial Intelligence-Generated Content (AIGC) has attracted significant attention, but two key challenges remain: maximizing subjective Quality of Experience (QoE) and improving energy efficiency, which are particularly pronounced in widely adopted Generative Diffusion Model (GDM)-based image generation services. In this paper, we propose a novel user-centric Interactive AI (IAI) approach for service management, with a distributed GDM-based AIGC framework that emphasizes efficient and cooperative deployment. The proposed method restructures the GDM inference process by allowing users with semantically similar prompts to share parts of the denoising chain. Furthermore, to maximize the users' subjective QoE, we propose an IAI approach, i.e., Reinforcement Learning With Large Language Models Interaction (RLLI), which utilizes Large Language Model (LLM)-empowered generative agents to replicate user interaction, providing real-time and subjective QoE feedback aligned with diverse user personalities. Lastly, we present the GDM-based Deep Deterministic Policy Gradient (GDDPG) algorithm, adapted to the proposed RLLI framework, to allocate communication and computing resources effectively while accounting for subjective user traits and dynamic wireless conditions. Simulation results demonstrate that G-DDPG improves total QoE by 15% compared with the standard DDPG algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16563v1">Enhancing LLM-Based Agents via Global Planning and Hierarchical Execution</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Intelligent agent systems based on Large Language Models (LLMs) have shown great potential in real-world applications. However, existing agent frameworks still face critical limitations in task planning and execution, restricting their effectiveness and generalizability. Specifically, current planning methods often lack clear global goals, leading agents to get stuck in local branches, or produce non-executable plans. Meanwhile, existing execution mechanisms struggle to balance complexity and stability, and their limited action space restricts their ability to handle diverse real-world tasks. To address these limitations, we propose GoalAct, a novel agent framework that introduces a continuously updated global planning mechanism and integrates a hierarchical execution strategy. GoalAct decomposes task execution into high-level skills, including searching, coding, writing and more, thereby reducing planning complexity while enhancing the agents' adaptability across diverse task scenarios. We evaluate GoalAct on LegalAgentBench, a benchmark with multiple types of legal tasks that require the use of multiple types of tools. Experimental results demonstrate that GoalAct achieves state-of-the-art (SOTA) performance, with an average improvement of 12.22% in success rate. These findings highlight GoalAct's potential to drive the development of more advanced intelligent agent systems, making them more effective across complex real-world applications. Our code can be found at https://github.com/cjj826/GoalAct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16511v1">QuaDMix: Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Quality and diversity are two critical metrics for the training data of large language models (LLMs), positively impacting performance. Existing studies often optimize these metrics separately, typically by first applying quality filtering and then adjusting data proportions. However, these approaches overlook the inherent trade-off between quality and diversity, necessitating their joint consideration. Given a fixed training quota, it is essential to evaluate both the quality of each data point and its complementary effect on the overall dataset. In this paper, we introduce a unified data selection framework called QuaDMix, which automatically optimizes the data distribution for LLM pretraining while balancing both quality and diversity. Specifically, we first propose multiple criteria to measure data quality and employ domain classification to distinguish data points, thereby measuring overall diversity. QuaDMix then employs a unified parameterized data sampling function that determines the sampling probability of each data point based on these quality and diversity related labels. To accelerate the search for the optimal parameters involved in the QuaDMix framework, we conduct simulated experiments on smaller models and use LightGBM for parameters searching, inspired by the RegMix method. Our experiments across diverse models and datasets demonstrate that QuaDMix achieves an average performance improvement of 7.2% across multiple benchmarks. These results outperform the independent strategies for quality and diversity, highlighting the necessity and ability to balance data quality and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16504v1">Intelligent Depression Prevention via LLM-Based Dialogue Analysis: Overcoming the Limitations of Scale-Dependent Diagnosis through Precise Emotional Pattern Recognition</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Existing depression screening predominantly relies on standardized questionnaires (e.g., PHQ-9, BDI), which suffer from high misdiagnosis rates (18-34% in clinical studies) due to their static, symptom-counting nature and susceptibility to patient recall bias. This paper presents an AI-powered depression prevention system that leverages large language models (LLMs) to analyze real-time conversational cues--including subtle emotional expressions (e.g., micro-sentiment shifts, self-referential language patterns)--for more accurate and dynamic mental state assessment. Our system achieves three key innovations: (1) Continuous monitoring through natural dialogue, detecting depression-indicative linguistic features (anhedonia markers, hopelessness semantics) with 89% precision (vs. 72% for PHQ-9); (2) Adaptive risk stratification that updates severity levels based on conversational context, reducing false positives by 41% compared to scale-based thresholds; and (3) Personalized intervention strategies tailored to users' emotional granularity, demonstrating 2.3x higher adherence rates than generic advice. Clinical validation with 450 participants shows the system identifies 92% of at-risk cases missed by traditional scales, while its explainable AI interface bridges the gap between automated analysis and clinician judgment. This work establishes conversational AI as a paradigm shift from episodic scale-dependent diagnosis to continuous, emotionally intelligent mental health monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16489v1">Amplified Vulnerabilities: Structured Jailbreak Attacks on LLM-based Multi-Agent Debate</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 33 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Multi-Agent Debate (MAD), leveraging collaborative interactions among Large Language Models (LLMs), aim to enhance reasoning capabilities in complex tasks. However, the security implications of their iterative dialogues and role-playing characteristics, particularly susceptibility to jailbreak attacks eliciting harmful content, remain critically underexplored. This paper systematically investigates the jailbreak vulnerabilities of four prominent MAD frameworks built upon leading commercial LLMs (GPT-4o, GPT-4, GPT-3.5-turbo, and DeepSeek) without compromising internal agents. We introduce a novel structured prompt-rewriting framework specifically designed to exploit MAD dynamics via narrative encapsulation, role-driven escalation, iterative refinement, and rhetorical obfuscation. Our extensive experiments demonstrate that MAD systems are inherently more vulnerable than single-agent setups. Crucially, our proposed attack methodology significantly amplifies this fragility, increasing average harmfulness from 28.14% to 80.34% and achieving attack success rates as high as 80% in certain scenarios. These findings reveal intrinsic vulnerabilities in MAD architectures and underscore the urgent need for robust, specialized defenses prior to real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05000v2">Needle Threading: Can LLMs Follow Threads through Near-Million-Scale Haystacks?</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Accepted at ICLR 2025
    </div>
    <details class="paper-abstract">
      As the context limits of Large Language Models (LLMs) increase, the range of possible applications and downstream functions broadens. In many real-world tasks, decisions depend on details scattered across collections of often disparate documents containing mostly irrelevant information. Long-context LLMs appear well-suited to this form of complex information retrieval and reasoning, which has traditionally proven costly and time-consuming. However, although the development of longer context models has seen rapid gains in recent years, our understanding of how effectively LLMs use their context has not kept pace. To address this, we conduct a set of retrieval experiments designed to evaluate the capabilities of 17 leading LLMs, such as their ability to follow threads of information through the context window. Strikingly, we find that many models are remarkably threadsafe: capable of simultaneously following multiple threads without significant loss in performance. Still, for many models, we find the effective context limit is significantly shorter than the supported context length, with accuracy decreasing as the context window grows. Our study also highlights the important point that token counts from different tokenizers should not be directly compared -- they often correspond to substantially different numbers of written characters. We release our code and long-context experimental data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16472v1">Harden and Catch for Just-in-Time Assured LLM-Based Software Testing: Open Research Challenges</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 To Appear as keynote paper at FSE 2025
    </div>
    <details class="paper-abstract">
      Despite decades of research and practice in automated software testing, several fundamental concepts remain ill-defined and under-explored, yet offer enormous potential real-world impact. We show that these concepts raise exciting new challenges in the context of Large Language Models for software test generation. More specifically, we formally define and investigate the properties of hardening and catching tests. A hardening test is one that seeks to protect against future regressions, while a catching test is one that catches such a regression or a fault in new functionality introduced by a code change. Hardening tests can be generated at any time and may become catching tests when a future regression is caught. We also define and motivate the Catching `Just-in-Time' (JiTTest) Challenge, in which tests are generated `just-in-time' to catch new faults before they land into production. We show that any solution to Catching JiTTest generation can also be repurposed to catch latent faults in legacy code. We enumerate possible outcomes for hardening and catching tests and JiTTests, and discuss open research problems, deployment options, and initial results from our work on automated LLM-based hardening at Meta. This paper\footnote{Author order is alphabetical. The corresponding author is Mark Harman.} was written to accompany the keynote by the authors at the ACM International Conference on the Foundations of Software Engineering (FSE) 2025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15585v1">A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10834v3">MetaLLM: A High-performant and Cost-efficient Dynamic Framework for Wrapping LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      The rapid progress in machine learning (ML) has brought forth many large language models (LLMs) that excel in various tasks and areas. These LLMs come with different abilities and costs in terms of computation or pricing. Since the demand for each query can vary, e.g., because of the queried domain or its complexity, defaulting to one LLM in an application is not usually the best choice, whether it is the biggest, priciest, or even the one with the best average test performance. Consequently, picking the right LLM that is both accurate and cost-effective for an application is necessary yet remains a challenge. In this paper, we introduce MetaLLM, a framework that dynamically and intelligently routes each query to the optimal LLM (among several available LLMs) for classification and multi-choice question-answering tasks, achieving significantly improved accuracy and cost-effectiveness. By framing the selection problem as a multi-armed bandit, MetaLLM balances prediction accuracy and cost efficiency under uncertainty. Our experiments, conducted on popular LLM platforms such as OpenAI and Together AI, as well as open-source LLM, showcase MetaLLM's efficacy in real-world scenarios, laying the groundwork for future extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15564v1">A Large-scale Class-level Benchmark Dataset for Code Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 This paper was submitted to the 29th International Conference on Evaluation and Assessment in Software Engineering (EASE 2025) AI models/data track
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated promising capabilities in code generation tasks. However, most existing benchmarks focus on isolated functions and fail to capture the complexity of real-world, class-level software structures. To address this gap, we introduce a large-scale, Python class-level dataset curated from $13{,}174$ real-world open-source projects. The dataset contains over 842,000 class skeletons, each including class and method signatures, along with associated docstrings when available. We preserve structural and contextual dependencies critical to realistic software development scenarios and enrich the dataset with static code metrics to support downstream analysis. To evaluate the usefulness of this dataset, we use extracted class skeletons as prompts for GPT-4 to generate full class implementations. Results show that the LLM-generated classes exhibit strong lexical and structural similarity to human-written counterparts, with average ROUGE@L, BLEU, and TSED scores of 0.80, 0.59, and 0.73, respectively. These findings confirm that well-structured prompts derived from real-world class skeletons significantly enhance LLM performance in class-level code generation. This dataset offers a valuable resource for benchmarking, training, and improving LLMs in realistic software engineering contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10284v2">Can LLMs Generate Tabular Summaries of Science Papers? Rethinking the Evaluation Protocol</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Literature review tables are essential for summarizing and comparing collections of scientific papers. We explore the task of generating tables that best fulfill a user's informational needs given a collection of scientific papers. Building on recent work (Newman et al., 2024), we extend prior approaches to address real-world complexities through a combination of LLM-based methods and human annotations. Our contributions focus on three key challenges encountered in real-world use: (i) User prompts are often under-specified; (ii) Retrieved candidate papers frequently contain irrelevant content; and (iii) Task evaluation should move beyond shallow text similarity techniques and instead assess the utility of inferred tables for information-seeking tasks (e.g., comparing papers). To support reproducible evaluation, we introduce ARXIV2TABLE, a more realistic and challenging benchmark for this task, along with a novel approach to improve literature review table generation in real-world scenarios. Our extensive experiments on this benchmark show that both open-weight and proprietary LLMs struggle with the task, highlighting its difficulty and the need for further advancements. Our dataset and code are available at https://github.com/JHU-CLSP/arXiv2Table.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15548v1">LLM-based Semantic Augmentation for Harmful Content Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated strong performance on simple text classification tasks, frequently under zero-shot settings. However, their efficacy declines when tackling complex social media challenges such as propaganda detection, hateful meme classification, and toxicity identification. Much of the existing work has focused on using LLMs to generate synthetic training data, overlooking the potential of LLM-based text preprocessing and semantic augmentation. In this paper, we introduce an approach that prompts LLMs to clean noisy text and provide context-rich explanations, thereby enhancing training sets without substantial increases in data volume. We systematically evaluate on the SemEval 2024 multi-label Persuasive Meme dataset and further validate on the Google Jigsaw toxic comments and Facebook hateful memes datasets to assess generalizability. Our results reveal that zero-shot LLM classification underperforms on these high-context tasks compared to supervised models. In contrast, integrating LLM-based semantic augmentation yields performance on par with approaches that rely on human-annotated data, at a fraction of the cost. These findings underscore the importance of strategically incorporating LLMs into machine learning (ML) pipeline for social media classification tasks, offering broad implications for combating harmful content online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15546v1">A Framework for Testing and Adapting REST APIs as LLM Tools</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are enabling autonomous agents to perform complex workflows using external tools or functions, often provided via REST APIs in enterprise systems. However, directly utilizing these APIs as tools poses challenges due to their complex input schemas, elaborate responses, and often ambiguous documentation. Current benchmarks for tool testing do not adequately address these complexities, leading to a critical gap in evaluating API readiness for agent-driven automation. In this work, we present a novel testing framework aimed at evaluating and enhancing the readiness of REST APIs to function as tools for LLM-based agents. Our framework transforms apis as tools, generates comprehensive test cases for the APIs, translates tests cases into natural language instructions suitable for agents, enriches tool definitions and evaluates the agent's ability t correctly invoke the API and process its inputs and responses. To provide actionable insights, we analyze the outcomes of 750 test cases, presenting a detailed taxonomy of errors, including input misinterpretation, output handling inconsistencies, and schema mismatches. Additionally, we classify these test cases to streamline debugging and refinement of tool integrations. This work offers a foundational step toward enabling enterprise APIs as tools, improving their usability in agent-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15544v1">llm-jp-modernbert: A ModernBERT Model Trained on a Large-Scale Japanese Corpus with Long Context Length</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Encoder-only transformer models like BERT are widely adopted as a pre-trained backbone for tasks like sentence classification and retrieval. However, pretraining of encoder models with large-scale corpora and long contexts has been relatively underexplored compared to decoder-only transformers. In this work, we present llm-jp-modernbert, a ModernBERT model trained on a publicly available, massive Japanese corpus with a context length of 8192 tokens. While our model does not surpass existing baselines on downstream tasks, it achieves good results on fill-mask test evaluations. We also analyze the effect of context length expansion through pseudo-perplexity experiments. Furthermore, we investigate sentence embeddings in detail, analyzing their transitions during training and comparing them with those from other existing models, confirming similar trends with models sharing the same architecture. To support reproducibility and foster the development of long-context BERT, we release our model, along with the training and evaluation code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10150v2">HistLLM: A Unified Framework for LLM-Based Multimodal Recommendation with User History Encoding and Compression</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 We want to withdraw this paper and revise its experimental details. The revised version will be uploaded after further verification
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have proven effective in leveraging textual data for recommendations, their application to multimodal recommendation tasks remains relatively underexplored. Although LLMs can process multimodal information through projection functions that map visual features into their semantic space, recommendation tasks often require representing users' history interactions through lengthy prompts combining text and visual elements, which not only hampers training and inference efficiency but also makes it difficult for the model to accurately capture user preferences from complex and extended prompts, leading to reduced recommendation performance. To address this challenge, we introduce HistLLM, an innovative multimodal recommendation framework that integrates textual and visual features through a User History Encoding Module (UHEM), compressing multimodal user history interactions into a single token representation, effectively facilitating LLMs in processing user preferences. Extensive experiments demonstrate the effectiveness and efficiency of our proposed mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12867v3">EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Human speech goes beyond the mere transfer of information; it is a profound exchange of emotions and a connection between individuals. While Text-to-Speech (TTS) models have made huge progress, they still face challenges in controlling the emotional expression in the generated speech. In this work, we propose EmoVoice, a novel emotion-controllable TTS model that exploits large language models (LLMs) to enable fine-grained freestyle natural language emotion control, and a phoneme boost variant design that makes the model output phoneme tokens and audio tokens in parallel to enhance content consistency, inspired by chain-of-thought (CoT) and chain-of-modality (CoM) techniques. Besides, we introduce EmoVoice-DB, a high-quality 40-hour English emotion dataset featuring expressive speech and fine-grained emotion labels with natural language descriptions. EmoVoice achieves state-of-the-art performance on the English EmoVoice-DB test set using only synthetic training data, and on the Chinese Secap test set using our in-house data. We further investigate the reliability of existing emotion evaluation metrics and their alignment with human perceptual preferences, and explore using SOTA multimodal LLMs GPT-4o-audio and Gemini to assess emotional speech. Demo samples are available at https://yanghaha0908.github.io/EmoVoice/. Dataset, code, and checkpoints will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15509v1">SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Simultaneous speech translation (SST) outputs translations in parallel with streaming speech input, balancing translation quality and latency. While large language models (LLMs) have been extended to handle the speech modality, streaming remains challenging as speech is prepended as a prompt for the entire generation process. To unlock LLM streaming capability, this paper proposes SimulS2S-LLM, which trains speech LLMs offline and employs a test-time policy to guide simultaneous inference. SimulS2S-LLM alleviates the mismatch between training and inference by extracting boundary-aware speech prompts that allows it to be better matched with text input data. SimulS2S-LLM achieves simultaneous speech-to-speech translation (Simul-S2ST) by predicting discrete output speech tokens and then synthesising output speech using a pre-trained vocoder. An incremental beam search is designed to expand the search space of speech token prediction without increasing latency. Experiments on the CVSS speech data show that SimulS2S-LLM offers a better translation quality-latency trade-off than existing methods that use the same training data, such as improving ASR-BLEU scores by 3 points at similar latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16273v1">Investigating LLMs in Clinical Triage: Promising Capabilities, Persistent Intersectional Biases</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 Accepted to GenAI4Health Workshop @ AAAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in clinical decision support, yet their application to triage remains underexplored. We systematically investigate the capabilities of LLMs in emergency department triage through two key dimensions: (1) robustness to distribution shifts and missing data, and (2) counterfactual analysis of intersectional biases across sex and race. We assess multiple LLM-based approaches, ranging from continued pre-training to in-context learning, as well as machine learning approaches. Our results indicate that LLMs exhibit superior robustness, and we investigate the key factors contributing to the promising LLM-based approaches. Furthermore, in this setting, we identify gaps in LLM preferences that emerge in particular intersections of sex and race. LLMs generally exhibit sex-based differences, but they are most pronounced in certain racial groups. These findings suggest that LLMs encode demographic preferences that may emerge in specific clinical contexts or particular combinations of characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16266v1">TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on edge platforms is challenged by their high computational and memory demands. Although recent low-bit quantization methods (e.g., BitNet, DeepSeek) compress weights to as little as 1.58 bits with minimal accuracy loss, edge deployment is still constrained by limited on-chip resources, power budgets, and the often-neglected latency of the prefill phase. We present TeLLMe, the first ternary LLM accelerator for low-power FPGAs (e.g., AMD KV260) that fully supports both prefill and autoregressive decoding using 1.58-bit weights and 8-bit activations. Our contributions include: (1) a table-lookup matrix engine for ternary matmul that merges grouped activations with online precomputation to minimize resource use; (2) a fused, bandwidth-efficient attention module featuring a reversed reordering scheme to accelerate prefill; and (3) a tightly integrated normalization and quantization--dequantization unit optimized for ultra-low-bit inference. Under a 7W power budget, TeLLMe delivers up to 9 tokens/s throughput over 1,024-token contexts and prefill latencies of 0.55--1.15 s for 64--128 token prompts, marking a significant energy-efficiency advance and establishing a new edge FPGA benchmark for generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17604v4">OmniScience: A Domain-Specialized LLM for Scientific Reasoning and Discovery</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in advancing scientific knowledge and addressing complex challenges. In this work, we introduce OmniScience, a specialized large reasoning model for general science, developed through three key components: (1) domain adaptive pretraining on a carefully curated corpus of scientific literature, (2) instruction tuning on a specialized dataset to guide the model in following domain-specific tasks, and (3) reasoning-based knowledge distillation through fine-tuning to significantly enhance its ability to generate contextually relevant and logically sound responses. We demonstrate the versatility of OmniScience by developing a battery agent that efficiently ranks molecules as potential electrolyte solvents or additives. Comprehensive evaluations reveal that OmniScience is competitive with state-of-the-art large reasoning models on the GPQA Diamond and domain-specific battery benchmarks, while outperforming all public reasoning and non-reasoning models with similar parameter counts. We further demonstrate via ablation experiments that domain adaptive pretraining and reasoning-based knowledge distillation are critical to attain our performance levels, across benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08619v4">Analyzing 16,193 LLM Papers for Fun and Profits</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping the landscape of computer science research, driving significant shifts in research priorities across diverse conferences and fields. This study provides a comprehensive analysis of the publication trend of LLM-related papers in 77 top-tier computer science conferences over the past six years (2019-2024). We approach this analysis from four distinct perspectives: (1) We investigate how LLM research is driving topic shifts within major conferences. (2) We adopt a topic modeling approach to identify various areas of LLM-related topic growth and reveal the topics of concern at different conferences. (3) We explore distinct contribution patterns of academic and industrial institutions. (4) We study the influence of national origins on LLM development trajectories. Synthesizing the findings from these diverse analytical angles, we derive ten key insights that illuminate the dynamics and evolution of the LLM research ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13657v2">Why Do Multi-Agent LLM Systems Fail?</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 ArXiv v2
    </div>
    <details class="paper-abstract">
      Despite growing enthusiasm for Multi-Agent LLM Systems (MAS), their performance gains on popular benchmarks often remain minimal compared with single-agent frameworks. This gap highlights the need to systematically analyze the challenges hindering MAS effectiveness. We present MAST (Multi-Agent System Failure Taxonomy), the first empirically grounded taxonomy designed to understand MAS failures. We analyze seven popular MAS frameworks across over 200 tasks, involving six expert human annotators. Through this process, we identify 14 unique failure modes, organized into 3 overarching categories, (i) specification issues, (ii) inter-agent misalignment, and (iii) task verification. MAST emerges iteratively from rigorous inter-annotator agreement studies, achieving a Cohen's Kappa score of 0.88. To support scalable evaluation, we develop a validated LLM-as-a-Judge pipeline integrated with MAST. We leverage two case studies to demonstrate MAST's practical utility in analyzing failures and guiding MAS development. Our findings reveal that identified failures require more complex solutions, highlighting a clear roadmap for future research. We open source our comprehensive dataset and LLM annotator to facilitate further development of MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16155v1">DATETIME: A new benchmark to measure LLM translation and reasoning capabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      This paper introduces DATETIME, a new high-quality benchmark designed to evaluate the translation and reasoning abilities of a Large Language Model (LLM) on datetimes. A datetime is simply a date and a time, for example '11th.february.2023 ,1:12:31'. Datetimes are an interesting domain because they are intuitive and straightforward for humans to process but present significant challenges for LLMs. At the time of writing, no publicly available benchmark exists for systematically evaluating LLMs on datetime processing. Our experiments show that state-of-the-art models exhibit significant difficulty with tasks involving reasoning on datetimes, and that General Artificial Intelligence is still a distant aspiration. We hypothesize that working with datetimes necessitates translation and/or computation capabilities, and the tasks of the benchmark are organized accordingly. Significant dispersion in performance across models is observed with surprisingly poor performance even on apparently trivial tasks. Whilst frontier models such as ChatGPT, Claude and Llama3.1 have evidently been built and trained with datetime reasoning abilities, significant improvement is required for the open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16144v1">Detecting Actionable Requests and Offers on Social Media During Crises Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Natural disasters often result in a surge of social media activity, including requests for assistance, offers of help, sentiments, and general updates. To enable humanitarian organizations to respond more efficiently, we propose a fine-grained hierarchical taxonomy to systematically organize crisis-related information about requests and offers into three critical dimensions: supplies, emergency personnel, and actions. Leveraging the capabilities of Large Language Models (LLMs), we introduce Query-Specific Few-shot Learning (QSF Learning) that retrieves class-specific labeled examples from an embedding database to enhance the model's performance in detecting and classifying posts. Beyond classification, we assess the actionability of messages to prioritize posts requiring immediate attention. Extensive experiments demonstrate that our approach outperforms baseline prompting strategies, effectively identifying and prioritizing actionable requests and offers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16078v1">LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      The success of Large Language Models (LLMs) has sparked interest in various agentic applications. A key hypothesis is that LLMs, leveraging common sense and Chain-of-Thought (CoT) reasoning, can effectively explore and efficiently solve complex domains. However, LLM agents have been found to suffer from sub-optimal exploration and the knowing-doing gap, the inability to effectively act on knowledge present in the model. In this work, we systematically study why LLMs perform sub-optimally in decision-making scenarios. In particular, we closely examine three prevalent failure modes: greediness, frequency bias, and the knowing-doing gap. We propose mitigation of these shortcomings by fine-tuning via Reinforcement Learning (RL) on self-generated CoT rationales. Our experiments across multi-armed bandits, contextual bandits, and Tic-tac-toe, demonstrate that RL fine-tuning enhances the decision-making abilities of LLMs by increasing exploration and narrowing the knowing-doing gap. Finally, we study both classic exploration mechanisms, such as $\epsilon$-greedy, and LLM-specific approaches, such as self-correction and self-consistency, to enable more effective fine-tuning of LLMs for decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16046v1">Certified Mitigation of Worst-Case LLM Copyright Infringement</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      The exposure of large language models (LLMs) to copyrighted material during pre-training raises concerns about unintentional copyright infringement post deployment. This has driven the development of "copyright takedown" methods, post-training approaches aimed at preventing models from generating content substantially similar to copyrighted ones. While current mitigation approaches are somewhat effective for average-case risks, we demonstrate that they overlook worst-case copyright risks exhibits by the existence of long, verbatim quotes from copyrighted sources. We propose BloomScrub, a remarkably simple yet highly effective inference-time approach that provides certified copyright takedown. Our method repeatedly interleaves quote detection with rewriting techniques to transform potentially infringing segments. By leveraging efficient data sketches (Bloom filters), our approach enables scalable copyright screening even for large-scale real-world corpora. When quotes beyond a length threshold cannot be removed, the system can abstain from responding, offering certified risk reduction. Experimental results show that BloomScrub reduces infringement risk, preserves utility, and accommodates different levels of enforcement stringency with adaptive abstention. Our results suggest that lightweight, inference-time methods can be surprisingly effective for copyright prevention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16032v1">LLMs meet Federated Learning for Scalable and Secure IoT Management</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 This work has been submitted to the IEEE Global Communications Conference (GLOBECOM) 2025 for possible publication
    </div>
    <details class="paper-abstract">
      The rapid expansion of IoT ecosystems introduces severe challenges in scalability, security, and real-time decision-making. Traditional centralized architectures struggle with latency, privacy concerns, and excessive resource consumption, making them unsuitable for modern large-scale IoT deployments. This paper presents a novel Federated Learning-driven Large Language Model (FL-LLM) framework, designed to enhance IoT system intelligence while ensuring data privacy and computational efficiency. The framework integrates Generative IoT (GIoT) models with a Gradient Sensing Federated Strategy (GSFS), dynamically optimizing model updates based on real-time network conditions. By leveraging a hybrid edge-cloud processing architecture, our approach balances intelligence, scalability, and security in distributed IoT environments. Evaluations on the IoT-23 dataset demonstrate that our framework improves model accuracy, reduces response latency, and enhances energy efficiency, outperforming traditional FL techniques (i.e., FedAvg, FedOpt). These findings highlight the potential of integrating LLM-powered federated learning into large-scale IoT ecosystems, paving the way for more secure, scalable, and adaptive IoT management solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16030v1">LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 CVPR 2025. If any references are missing, please contact joyachen@u.nus.edu
    </div>
    <details class="paper-abstract">
      Recent video large language models (Video LLMs) often depend on costly human annotations or proprietary model APIs (e.g., GPT-4o) to produce training data, which limits their training at scale. In this paper, we explore large-scale training for Video LLM with cheap automatic speech recognition (ASR) transcripts. Specifically, we propose a novel streaming training approach that densely interleaves the ASR words and video frames according to their timestamps. Compared to previous studies in vision-language representation with ASR, our method naturally fits the streaming characteristics of ASR, thus enabling the model to learn temporally-aligned, fine-grained vision-language modeling. To support the training algorithm, we introduce a data production pipeline to process YouTube videos and their closed captions (CC, same as ASR), resulting in Live-CC-5M dataset for pre-training and Live-WhisperX-526K dataset for high-quality supervised fine-tuning (SFT). Remarkably, even without SFT, the ASR-only pre-trained LiveCC-7B-Base model demonstrates competitive general video QA performance and exhibits a new capability in real-time video commentary. To evaluate this, we carefully design a new LiveSports-3K benchmark, using LLM-as-a-judge to measure the free-form commentary. Experiments show our final LiveCC-7B-Instruct model can surpass advanced 72B models (Qwen2.5-VL-72B-Instruct, LLaVA-Video-72B) in commentary quality even working in a real-time mode. Meanwhile, it achieves state-of-the-art results at the 7B/8B scale on popular video QA benchmarks such as VideoMME and OVOBench, demonstrating the broad generalizability of our approach. All resources of this paper have been released at https://showlab.github.io/livecc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16027v1">Benchmarking LLM for Code Smells Detection: OpenAI GPT-4.0 vs DeepSeek-V3</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Determining the most effective Large Language Model for code smell detection presents a complex challenge. This study introduces a structured methodology and evaluation matrix to tackle this issue, leveraging a curated dataset of code samples consistently annotated with known smells. The dataset spans four prominent programming languages Java, Python, JavaScript, and C++; allowing for cross language comparison. We benchmark two state of the art LLMs, OpenAI GPT 4.0 and DeepSeek-V3, using precision, recall, and F1 score as evaluation metrics. Our analysis covers three levels of detail: overall performance, category level performance, and individual code smell type performance. Additionally, we explore cost effectiveness by comparing the token based detection approach of GPT 4.0 with the pattern-matching techniques employed by DeepSeek V3. The study also includes a cost analysis relative to traditional static analysis tools such as SonarQube. The findings offer valuable guidance for practitioners in selecting an efficient, cost effective solution for automated code smell detection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03621v2">Network-aided Efficient LLM Services With Denoising-inspired Prompt Compression</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 This paper has been accepted by SCIENCE CHINA Information Sciences. arXiv admin note: substantial text overlap with arXiv:2411.18010
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, leading to their increasing adoption in diverse services delivered through wireless networks. There is a growing trend toward longer prompts to better leverage LLMs' capabilities and address difficult tasks. However, longer prompts not only increase data transmission costs but also require more computing resources and processing time, which impacts overall system efficiency and user experience. To address this challenge, we propose Joint Power and Prompt Optimization (JPPO), a framework that combines Small Language Model (SLM)-based prompt compression with wireless power allocation optimization. By deploying SLM at edge devices for prompt compression and employing Deep Reinforcement Learning (DRL) for joint optimization of compression ratio and transmission power, JPPO effectively balances service quality with resource efficiency. Furthermore, inspired by denoising diffusion models, we design a denoising-inspired prompt compression approach that iteratively compresses prompts by gradually removing non-critical information, further enhancing the framework's performance. Experimental results with long prompt tokens demonstrate that our framework achieves high service fidelity while optimizing power usage in wireless LLM services, significantly reducing the total service response time. With our DRL-based JPPO, the framework maintains fidelity comparable to the no-compression baseline while still achieving a 17% service time reduction through adaptive compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14489v2">Optimizing SLO-oriented LLM Serving with PD-Multiplexing</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Modern LLM services demand high throughput and stringent SLO guarantees across two distinct inference phases-prefill and decode-and complex multi-turn workflows. However, current systems face a fundamental tradeoff: out-of-place compute partition enables per-phase SLO attainment, while in-place memory sharing maximizes throughput via KV cache reuse. Moreover, existing in-place compute partition also encounters low utilization and high overhead due to phase-coupling design. We present Drift, a new LLM serving framework that resolves this tension via PD multiplexing, enabling in-place and phase-decoupled compute partition. Drift leverages low-level GPU partitioning techniques to multiplex prefill and decode phases spatially and adaptively on shared GPUs, while preserving in-place memory sharing. To fully leverage the multiplexing capability, Drift introduces an adaptive gang scheduling mechanism, a contention-free modeling method, and a SLO-aware dispatching policy. Evaluation shows that Drift achieves an average $5.1\times$ throughput improvement (up to $17.5\times$) over state-of-the-art baselines, while consistently meeting SLO targets under complex LLM workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15965v1">From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 26 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Memory is the process of encoding, storing, and retrieving information, allowing humans to retain experiences, knowledge, skills, and facts over time, and serving as the foundation for growth and effective interaction with the world. It plays a crucial role in shaping our identity, making decisions, learning from past experiences, building relationships, and adapting to changes. In the era of large language models (LLMs), memory refers to the ability of an AI system to retain, recall, and use information from past interactions to improve future responses and interactions. Although previous research and reviews have provided detailed descriptions of memory mechanisms, there is still a lack of a systematic review that summarizes and analyzes the relationship between the memory of LLM-driven AI systems and human memory, as well as how we can be inspired by human memory to construct more powerful memory systems. To achieve this, in this paper, we propose a comprehensive survey on the memory of LLM-driven AI systems. In particular, we first conduct a detailed analysis of the categories of human memory and relate them to the memory of AI systems. Second, we systematically organize existing memory-related work and propose a categorization method based on three dimensions (object, form, and time) and eight quadrants. Finally, we illustrate some open problems regarding the memory of current AI systems and outline possible future directions for memory in the era of large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15930v1">StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become the core post-training technique for large language models (LLMs). RL for LLMs involves two stages: generation and training. The LLM first generates samples online, which are then used to derive rewards for training. The conventional view holds that the colocated architecture, where the two stages share resources via temporal multiplexing, outperforms the disaggregated architecture, in which dedicated resources are assigned to each stage. However, in real-world deployments, we observe that the colocated architecture suffers from resource coupling, where the two stages are constrained to use the same resources. This coupling compromises the scalability and cost-efficiency of colocated RL in large-scale training. In contrast, the disaggregated architecture allows for flexible resource allocation, supports heterogeneous training setups, and facilitates cross-datacenter deployment. StreamRL is designed with disaggregation from first principles and fully unlocks its potential by addressing two types of performance bottlenecks in existing disaggregated RL frameworks: pipeline bubbles, caused by stage dependencies, and skewness bubbles, resulting from long-tail output length distributions. To address pipeline bubbles, StreamRL breaks the traditional stage boundary in synchronous RL algorithms through stream generation and achieves full overlapping in asynchronous RL. To address skewness bubbles, StreamRL employs an output-length ranker model to identify long-tail samples and reduces generation time via skewness-aware dispatching and scheduling. Experiments show that StreamRL improves throughput by up to 2.66x compared to existing state-of-the-art systems, and improves cost-effectiveness by up to 1.33x in a heterogeneous, cross-datacenter setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09597v4">Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous tasks, yet principled explanations for their underlying mechanisms and several phenomena, such as scaling laws, hallucinations, and related behaviors, remain elusive. In this work, we revisit the classical relationship between compression and prediction, grounded in Kolmogorov complexity and Shannon information theory, to provide deeper insights into LLM behaviors. By leveraging the Kolmogorov Structure Function and interpreting LLM compression as a two-part coding process, we offer a detailed view of how LLMs acquire and store information across increasing model and data scales -- from pervasive syntactic patterns to progressively rarer knowledge elements. Motivated by this theoretical perspective and natural assumptions inspired by Heap's and Zipf's laws, we introduce a simplified yet representative hierarchical data-generation framework called the Syntax-Knowledge model. Under the Bayesian setting, we show that prediction and compression within this model naturally lead to diverse learning and scaling behaviors of LLMs. In particular, our theoretical analysis offers intuitive and principled explanations for both data and model scaling laws, the dynamics of knowledge acquisition during training and fine-tuning, factual knowledge hallucinations in LLMs. The experimental results validate our theoretical predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09385v2">AI Predicts AGI: Leveraging AGI Forecasting and Peer Review to Explore LLMs' Complex Reasoning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 47 pages, 8 figures, 17 tables, appendix with data and code
    </div>
    <details class="paper-abstract">
      We tasked 16 state-of-the-art large language models (LLMs) with estimating the likelihood of Artificial General Intelligence (AGI) emerging by 2030. To assess the quality of these forecasts, we implemented an automated peer review process (LLM-PR). The LLMs' estimates varied widely, ranging from 3% (Reka- Core) to 47.6% (GPT-4o), with a median of 12.5%. These estimates closely align with a recent expert survey that projected a 10% likelihood of AGI by 2027, underscoring the relevance of LLMs in forecasting complex, speculative scenarios. The LLM-PR process demonstrated strong reliability, evidenced by a high Intraclass Correlation Coefficient (ICC = 0.79), reflecting notable consistency in scoring across the models. Among the models, Pplx-70b-online emerged as the top performer, while Gemini-1.5-pro-api ranked the lowest. A cross-comparison with external benchmarks, such as LMSYS Chatbot Arena, revealed that LLM rankings remained consistent across different evaluation methods, suggesting that existing benchmarks may not encapsulate some of the skills relevant for AGI prediction. We further explored the use of weighting schemes based on external benchmarks, optimizing the alignment of LLMs' predictions with human expert forecasts. This analysis led to the development of a new, 'AGI benchmark' designed to highlight performance differences in AGI-related tasks. Our findings offer insights into LLMs' capabilities in speculative, interdisciplinary forecasting tasks and emphasize the growing need for innovative evaluation frameworks for assessing AI performance in complex, uncertain real-world scenarios.
    </details>
</div>
