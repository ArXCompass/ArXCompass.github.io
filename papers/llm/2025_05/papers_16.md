# llm - 2025_05

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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- Part 16
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04806v1">Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 7 Pages, 6 Figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04736v1">The Promise and Limits of LLMs in Constructing Proofs and Hints for Logic Problems in Intelligent Tutoring Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Intelligent tutoring systems have demonstrated effectiveness in teaching formal propositional logic proofs, but their reliance on template-based explanations limits their ability to provide personalized student feedback. While large language models (LLMs) offer promising capabilities for dynamic feedback generation, they risk producing hallucinations or pedagogically unsound explanations. We evaluated the stepwise accuracy of LLMs in constructing multi-step symbolic logic proofs, comparing six prompting techniques across four state-of-the-art LLMs on 358 propositional logic problems. Results show that DeepSeek-V3 achieved superior performance with 84.4% accuracy on stepwise proof construction and excelled particularly in simpler rules. We further used the best-performing LLM to generate explanatory hints for 1,050 unique student problem-solving states from a logic ITS and evaluated them on 4 criteria with both an LLM grader and human expert ratings on a 20% sample. Our analysis finds that LLM-generated hints were 75% accurate and rated highly by human evaluators on consistency and clarity, but did not perform as well explaining why the hint was provided or its larger context. Our results demonstrate that LLMs may be used to augment tutoring systems with logic tutoring hints, but requires additional modifications to ensure accuracy and pedagogical appropriateness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04732v1">QBD-RankedDataGen: Generating Custom Ranked Datasets for Improving Query-By-Document Search Using LLM-Reranking with Reduced Human Effort</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      The Query-By-Document (QBD) problem is an information retrieval problem where the query is a document, and the retrieved candidates are documents that match the query document, often in a domain or query specific manner. This can be crucial for tasks such as patent matching, legal or compliance case retrieval, and academic literature review. Existing retrieval methods, including keyword search and document embeddings, can be optimized with domain-specific datasets to improve QBD search performance. However, creating these domain-specific datasets is often costly and time-consuming. Our work introduces a process to generate custom QBD-search datasets and compares a set of methods to use in this problem, which we refer to as QBD-RankedDatagen. We provide a comparative analysis of our proposed methods in terms of cost, speed, and the human interface with the domain experts. The methods we compare leverage Large Language Models (LLMs) which can incorporate domain expert input to produce document scores and rankings, as well as explanations for human review. The process and methods for it that we present can significantly reduce human effort in dataset creation for custom domains while still obtaining sufficient expert knowledge for tuning retrieval models. We evaluate our methods on QBD datasets from the Text Retrieval Conference (TREC) and finetune the parameters of the BM25 model -- which is used in many industrial-strength search engines like OpenSearch -- using the generated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04723v1">SOAEsV2-7B/72B: Full-Pipeline Optimization for State-Owned Enterprise LLMs via Continual Pre-Training, Domain-Progressive SFT and Distillation-Enhanced Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      This study addresses key challenges in developing domain-specific large language models (LLMs) for Chinese state-owned assets and enterprises (SOAEs), where current approaches face three limitations: 1) constrained model capacity that limits knowledge integration and cross-task adaptability; 2) excessive reliance on domain-specific supervised fine-tuning (SFT) data, which neglects the broader applicability of general language patterns; and 3) inefficient inference acceleration for large models processing long contexts. In this work, we propose SOAEsV2-7B/72B, a specialized LLM series developed via a three-phase framework: 1) continual pre-training integrates domain knowledge while retaining base capabilities; 2) domain-progressive SFT employs curriculum-based learning strategy, transitioning from weakly relevant conversational data to expert-annotated SOAEs datasets to optimize domain-specific tasks; 3) distillation-enhanced speculative decoding accelerates inference via logit distillation between 72B target and 7B draft models, achieving 1.39-1.52$\times$ speedup without quality loss. Experimental results demonstrate that our domain-specific pre-training phase maintains 99.8% of original general language capabilities while significantly improving domain performance, resulting in a 1.08$\times$ improvement in Rouge-1 score and a 1.17$\times$ enhancement in BLEU-4 score. Ablation studies further show that domain-progressive SFT outperforms single-stage training, achieving 1.02$\times$ improvement in Rouge-1 and 1.06$\times$ in BLEU-4. Our work introduces a comprehensive, full-pipeline approach for optimizing SOAEs LLMs, bridging the gap between general language capabilities and domain-specific expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04710v1">Exploring Influence Factors on LLM Suitability for No-Code Development of End User IoT Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      With the increasing popularity of IoT applications, end users demand more personalized and intuitive functionality. A major obstacle for this, however, is that custom IoT functionality today still requires at least some coding skills. To address this, no-code development platforms have been proposed as a solution for empowering non-technical users to create applications. However, such platforms still require a certain level of technical expertise for structuring process steps or defining event-action relations. The advent of LLMs can further enhance no-code platforms by enabling natural language-based interaction, automating of complex tasks, and dynamic code generation. By allowing users to describe their requirements in natural language, LLMs can significantly streamline no-code development. As LLMs vary in performance, architecture, training data used, and the use cases they target, it is still unclear which models are best suited and what are the influence factors determining this fit. In particular, no-code development of IoT applications by non-technical users will have completely different demands on LLMs than, e.g., code generation for more open-ended applications or for supporting professional developers. In this paper, we explore the factors influencing the suitability of LLMs to no-code development of IoT applications. We also examine the role of input prompt language on accuracy and quality of generated applications as well as the influence of LLM training data. By conducting comprehensive experiments with a range of LLMs, we provide valuable insights for optimizing LLM-powered no-code platforms, guiding the selection of the suitable LLMs and their effective application. Our findings contribute to improving the accessibility, efficiency, and user experience of no-code IoT development, ultimately enabling broader adoption of IoT technologies among non-expert users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04673v1">REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 13 pages (8 main), to be published in IJCAI 2025
    </div>
    <details class="paper-abstract">
      Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o. We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03473v1">Evaluation of LLMs on Long-tail Entity Linking in Historical Documents</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Entity Linking (EL) plays a crucial role in Natural Language Processing (NLP) applications, enabling the disambiguation of entity mentions by linking them to their corresponding entries in a reference knowledge base (KB). Thanks to their deep contextual understanding capabilities, LLMs offer a new perspective to tackle EL, promising better results than traditional methods. Despite the impressive generalization capabilities of LLMs, linking less popular, long-tail entities remains challenging as these entities are often underrepresented in training data and knowledge bases. Furthermore, the long-tail EL task is an understudied problem, and limited studies address it with LLMs. In the present work, we assess the performance of two popular LLMs, GPT and LLama3, in a long-tail entity linking scenario. Using MHERCL v0.1, a manually annotated benchmark of sentences from domain-specific historical texts, we quantitatively compare the performance of LLMs in identifying and linking entities to their corresponding Wikidata entries against that of ReLiK, a state-of-the-art Entity Linking and Relation Extraction framework. Our preliminary experiments reveal that LLMs perform encouragingly well in long-tail EL, indicating that this technology can be a valuable adjunct in filling the gap between head and long-tail EL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04430v3">The Struggles of LLMs in Cross-lingual Code Clone Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 Accepted for publication at the ACM International Conference on the Foundations of Software Engineering (FSE) 2025
    </div>
    <details class="paper-abstract">
      With the involvement of multiple programming languages in modern software development, cross-lingual code clone detection has gained traction within the software engineering community. Numerous studies have explored this topic, proposing various promising approaches. Inspired by the significant advances in machine learning in recent years, particularly Large Language Models (LLMs), which have demonstrated their ability to tackle various tasks, this paper revisits cross-lingual code clone detection. We evaluate the performance of five (05) LLMs and eight prompts (08) for the identification of cross-lingual code clones. Additionally, we compare these results against two baseline methods. Finally, we evaluate a pre-trained embedding model to assess the effectiveness of the generated representations for classifying clone and non-clone pairs. The studies involving LLMs and Embedding models are evaluated using two widely used cross-lingual datasets, XLCoST and CodeNet. Our results show that LLMs can achieve high F1 scores, up to 0.99, for straightforward programming examples. However, they not only perform less well on programs associated with complex programming challenges but also do not necessarily understand the meaning of "code clones" in a cross-lingual setting. We show that embedding models used to represent code fragments from different programming languages in the same representation space enable the training of a basic classifier that outperforms all LLMs by ~1 and ~20 percentage points on the XLCoST and CodeNet datasets, respectively. This finding suggests that, despite the apparent capabilities of LLMs, embeddings provided by embedding models offer suitable representations to achieve state-of-the-art performance in cross-lingual code clone detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03434v1">Procedural Memory Is Not All You Need: Bridging Cognitive Gaps in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 Accepted to the workshop on Hybrid AI for Human-Centric Personalization (HyPer), co-located with ACM UMAP '25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) represent a landmark achievement in Artificial Intelligence (AI), demonstrating unprecedented proficiency in procedural tasks such as text generation, code completion, and conversational coherence. These capabilities stem from their architecture, which mirrors human procedural memory -- the brain's ability to automate repetitive, pattern-driven tasks through practice. However, as LLMs are increasingly deployed in real-world applications, it becomes impossible to ignore their limitations operating in complex, unpredictable environments. This paper argues that LLMs, while transformative, are fundamentally constrained by their reliance on procedural memory. To create agents capable of navigating ``wicked'' learning environments -- where rules shift, feedback is ambiguous, and novelty is the norm -- we must augment LLMs with semantic memory and associative learning systems. By adopting a modular architecture that decouples these cognitive functions, we can bridge the gap between narrow procedural expertise and the adaptive intelligence required for real-world problem-solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03406v1">Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      This research paper investigates the application of Large Language Models (LLMs) in healthcare, specifically focusing on enhancing medical decision support through Retrieval-Augmented Generation (RAG) integrated with hospital-specific data and fine-tuning using Quantized Low-Rank Adaptation (QLoRA). The system utilizes Llama 3.2-3B-Instruct as its foundation model. By embedding and retrieving context-relevant healthcare information, the system significantly improves response accuracy. QLoRA facilitates notable parameter efficiency and memory optimization, preserving the integrity of medical information through specialized quantization techniques. Our research also shows that our model performs relatively well on various medical benchmarks, indicating that it can be used to make basic medical suggestions. This paper details the system's technical components, including its architecture, quantization methods, and key healthcare applications such as enhanced disease prediction from patient symptoms and medical history, treatment suggestions, and efficient summarization of complex medical reports. We touch on the ethical considerations-patient privacy, data security, and the need for rigorous clinical validation-as well as the practical challenges of integrating such systems into real-world healthcare workflows. Furthermore, the lightweight quantized weights ensure scalability and ease of deployment even in low-resource hospital environments. Finally, the paper concludes with an analysis of the broader impact of LLMs on healthcare and outlines future directions for LLMs in medical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03345v1">Elevating Cyber Threat Intelligence against Disinformation Campaigns with LLM-based Concept Extraction and the FakeCTI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      The swift spread of fake news and disinformation campaigns poses a significant threat to public trust, political stability, and cybersecurity. Traditional Cyber Threat Intelligence (CTI) approaches, which rely on low-level indicators such as domain names and social media handles, are easily evaded by adversaries who frequently modify their online infrastructure. To address these limitations, we introduce a novel CTI framework that focuses on high-level, semantic indicators derived from recurrent narratives and relationships of disinformation campaigns. Our approach extracts structured CTI indicators from unstructured disinformation content, capturing key entities and their contextual dependencies within fake news using Large Language Models (LLMs). We further introduce FakeCTI, the first dataset that systematically links fake news to disinformation campaigns and threat actors. To evaluate the effectiveness of our CTI framework, we analyze multiple fake news attribution techniques, spanning from traditional Natural Language Processing (NLP) to fine-tuned LLMs. This work shifts the focus from low-level artifacts to persistent conceptual structures, establishing a scalable and adaptive approach to tracking and countering disinformation campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03336v1">Avoid Recommending Out-of-Domain Items: Constrained Generative Recommendation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise for generative recommender systems due to their transformative capabilities in user interaction. However, ensuring they do not recommend out-of-domain (OOD) items remains a challenge. We study two distinct methods to address this issue: RecLM-ret, a retrieval-based method, and RecLM-cgen, a constrained generation method. Both methods integrate seamlessly with existing LLMs to ensure in-domain recommendations. Comprehensive experiments on three recommendation datasets demonstrate that RecLM-cgen consistently outperforms RecLM-ret and existing LLM-based recommender models in accuracy while eliminating OOD recommendations, making it the preferred method for adoption. Additionally, RecLM-cgen maintains strong generalist capabilities and is a lightweight plug-and-play module for easy integration into LLMs, offering valuable practical benefits for the community. Source code is available at https://github.com/microsoft/RecAI
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03293v1">Ψ-Arena: Interactive Assessment and Optimization of LLM-based Psychological Counselors with Tripartite Feedback</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in providing scalable mental health support, while evaluating their counseling capability remains crucial to ensure both efficacy and safety. Existing evaluations are limited by the static assessment that focuses on knowledge tests, the single perspective that centers on user experience, and the open-loop framework that lacks actionable feedback. To address these issues, we propose {\Psi}-Arena, an interactive framework for comprehensive assessment and optimization of LLM-based counselors, featuring three key characteristics: (1) Realistic arena interactions that simulate real-world counseling through multi-stage dialogues with psychologically profiled NPC clients, (2) Tripartite evaluation that integrates assessments from the client, counselor, and supervisor perspectives, and (3) Closed-loop optimization that iteratively improves LLM counselors using diagnostic feedback. Experiments across eight state-of-the-art LLMs show significant performance variations in different real-world scenarios and evaluation perspectives. Moreover, reflection-based optimization results in up to a 141% improvement in counseling performance. We hope PsychoArena provides a foundational resource for advancing reliable and human-aligned LLM applications in mental healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10913v3">ASHABot: An LLM-Powered Chatbot to Support the Informational Needs of Community Health Workers</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Community health workers (CHWs) provide last-mile healthcare services but face challenges due to limited medical knowledge and training. This paper describes the design, deployment, and evaluation of ASHABot, an LLM-powered, experts-in-the-loop, WhatsApp-based chatbot to address the information needs of CHWs in India. Through interviews with CHWs and their supervisors and log analysis, we examine factors affecting their engagement with ASHABot, and ASHABot's role in addressing CHWs' informational needs. We found that ASHABot provided a private channel for CHWs to ask rudimentary and sensitive questions they hesitated to ask supervisors. CHWs trusted the information they received on ASHABot and treated it as an authoritative resource. CHWs' supervisors expanded their knowledge by contributing answers to questions ASHABot failed to answer, but were concerned about demands on their workload and increased accountability. We emphasize positioning LLMs as supplemental fallible resources within the community healthcare ecosystem, instead of as replacements for supervisor support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03275v1">RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle to effectively utilize a growing number of external tools, such as those defined by the Model Context Protocol (MCP)\cite{IntroducingMCP}, due to prompt bloat and selection complexity. We introduce RAG-MCP, a Retrieval-Augmented Generation framework that overcomes this challenge by offloading tool discovery. RAG-MCP uses semantic retrieval to identify the most relevant MCP(s) for a given query from an external index before engaging the LLM. Only the selected tool descriptions are passed to the model, drastically reducing prompt size and simplifying decision-making. Experiments, including an MCP stress test, demonstrate RAG-MCP significantly cuts prompt tokens (e.g., by over 50%) and more than triples tool selection accuracy (43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables scalable and accurate tool integration for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02579v2">EMORL: Ensemble Multi-Objective Reinforcement Learning for Efficient and Flexible LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 13 pages, 9 figures, submitted to SIGDIAL 2025 conference
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) for large language model (LLM) fine-tuning show promise in addressing multi-objective tasks but still face significant challenges, including complex objective balancing, low training efficiency, poor scalability, and limited explainability. Leveraging ensemble learning principles, we introduce an Ensemble Multi-Objective RL (EMORL) framework that fine-tunes multiple models with individual objectives while optimizing their aggregation after the training to improve efficiency and flexibility. Our method is the first to aggregate the last hidden states of individual models, incorporating contextual information from multiple objectives. This approach is supported by a hierarchical grid search algorithm that identifies optimal weighted combinations. We evaluate EMORL on counselor reflection generation tasks, using text-scoring LLMs to evaluate the generations and provide rewards during RL fine-tuning. Through comprehensive experiments on the PAIR and Psych8k datasets, we demonstrate the advantages of EMORL against existing baselines: significantly lower and more stable training consumption ($17,529\pm 1,650$ data points and $6,573\pm 147.43$ seconds), improved scalability and explainability, and comparable performance across multiple objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03196v1">A Trustworthy Multi-LLM Network: Challenges,Solutions, and A Use Case</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong potential across a variety of tasks in communications and networking due to their advanced reasoning capabilities. However, because different LLMs have different model structures and are trained using distinct corpora and methods, they may offer varying optimization strategies for the same network issues. Moreover, the limitations of an individual LLM's training data, aggravated by the potential maliciousness of its hosting device, can result in responses with low confidence or even bias. To address these challenges, we propose a blockchain-enabled collaborative framework that connects multiple LLMs into a Trustworthy Multi-LLM Network (MultiLLMN). This architecture enables the cooperative evaluation and selection of the most reliable and high-quality responses to complex network optimization problems. Specifically, we begin by reviewing related work and highlighting the limitations of existing LLMs in collaboration and trust, emphasizing the need for trustworthiness in LLM-based systems. We then introduce the workflow and design of the proposed Trustworthy MultiLLMN framework. Given the severity of False Base Station (FBS) attacks in B5G and 6G communication systems and the difficulty of addressing such threats through traditional modeling techniques, we present FBS defense as a case study to empirically validate the effectiveness of our approach. Finally, we outline promising future research directions in this emerging area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03179v1">Bridging Expertise Gaps: The Role of LLMs in Human-AI Collaboration for Cybersecurity</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 20 pages, 10 figures, 2 tables, under review
    </div>
    <details class="paper-abstract">
      This study investigates whether large language models (LLMs) can function as intelligent collaborators to bridge expertise gaps in cybersecurity decision-making. We examine two representative tasks-phishing email detection and intrusion detection-that differ in data modality, cognitive complexity, and user familiarity. Through a controlled mixed-methods user study, n = 58 (phishing, n = 34; intrusion, n = 24), we find that human-AI collaboration improves task performance,reducing false positives in phishing detection and false negatives in intrusion detection. A learning effect is also observed when participants transition from collaboration to independent work, suggesting that LLMs can support long-term skill development. Our qualitative analysis shows that interaction dynamics-such as LLM definitiveness, explanation style, and tone-influence user trust, prompting strategies, and decision revision. Users engaged in more analytic questioning and showed greater reliance on LLM feedback in high-complexity settings. These results provide design guidance for building interpretable, adaptive, and trustworthy human-AI teaming systems, and demonstrate that LLMs can meaningfully support non-experts in reasoning through complex cybersecurity problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03171v1">CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Neurosymbolic approaches integrating large language models with formal reasoning have recently achieved human-level performance on mathematics competition problems in algebra, geometry and number theory. In comparison, combinatorics remains a challenging domain, characterized by a lack of appropriate benchmarks and theorem libraries. To address this gap, we introduce CombiBench, a comprehensive benchmark comprising 100 combinatorial problems, each formalized in Lean~4 and paired with its corresponding informal statement. The problem set covers a wide spectrum of difficulty levels, ranging from middle school to IMO and university level, and span over ten combinatorial topics. CombiBench is suitable for testing IMO solving capabilities since it includes all IMO combinatorial problems since 2000 (except IMO 2004 P3 as its statement contain an images). Furthermore, we provide a comprehensive and standardized evaluation framework, dubbed Fine-Eval (for $\textbf{F}$ill-in-the-blank $\textbf{in}$ L$\textbf{e}$an Evaluation), for formal mathematics. It accommodates not only proof-based problems but also, for the first time, the evaluation of fill-in-the-blank questions. Using Fine-Eval as the evaluation method and Kimina Lean Server as the backend, we benchmark several LLMs on CombiBench and observe that their capabilities for formally solving combinatorial problems remain limited. Among all models tested (none of which has been trained for this particular task), Kimina-Prover attains the best results, solving 7 problems (out of 100) under both ``with solution'' and ``without solution'' scenarios. We open source the benchmark dataset alongside with the code of the proposed evaluation method at https://github.com/MoonshotAI/CombiBench/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03161v1">An LLM-based Self-Evolving Security Framework for 6G Space-Air-Ground Integrated Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Recently emerged 6G space-air-ground integrated networks (SAGINs), which integrate satellites, aerial networks, and terrestrial communications, offer ubiquitous coverage for various mobile applications. However, the highly dynamic, open, and heterogeneous nature of SAGINs poses severe security issues. Forming a defense line of SAGINs suffers from two preliminary challenges: 1) accurately understanding massive unstructured multi-dimensional threat information to generate defense strategies against various malicious attacks, 2) rapidly adapting to potential unknown threats to yield more effective security strategies. To tackle the above two challenges, we propose a novel security framework for SAGINs based on Large Language Models (LLMs), which consists of two key ingredients LLM-6GNG and 6G-INST. Our proposed LLM-6GNG leverages refined chain-of-thought (CoT) reasoning and dynamic multi-agent mechanisms to analyze massive unstructured multi-dimensional threat data and generate comprehensive security strategies, thus addressing the first challenge. Our proposed 6G-INST relies on a novel self-evolving method to automatically update LLM-6GNG, enabling it to accommodate unknown threats under dynamic communication environments, thereby addressing the second challenge. Additionally, we prototype the proposed framework with ns-3, OpenAirInterface (OAI), and software-defined radio (SDR). Experiments on three benchmarks demonstrate the effectiveness of our framework. The results show that our framework produces highly accurate security strategies that remain robust against a variety of unknown attacks. We will release our code to contribute to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08717v4">Social Opinions Prediction Utilizes Fusing Dynamics Equation with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      In the context where social media emerges as a pivotal platform for social movements and shaping public opinion, accurately simulating and predicting the dynamics of user opinions is of significant importance. Such insights are vital for understanding social phenomena, informing policy decisions, and guiding public opinion. Unfortunately, traditional algorithms based on idealized models and disregarding social data often fail to capture the complexity and nuance of real-world social interactions. This study proposes the Fusing Dynamics Equation-Large Language Model (FDE-LLM) algorithm. This innovative approach aligns the actions and evolution of opinions in Large Language Models (LLMs) with the real-world data on social networks. The FDE-LLM devides users into two roles: opinion leaders and followers. Opinion leaders use LLM for role-playing and employ Cellular Automata(CA) to constrain opinion changes. In contrast, opinion followers are integrated into a dynamic system that combines the CA model with the Susceptible-Infectious-Recovered (SIR) model. This innovative design significantly improves the accuracy of the simulation. Our experiments utilized four real-world datasets from Weibo. The result demonstrates that the FDE-LLM significantly outperforms traditional Agent-Based Modeling (ABM) algorithms and LLM-based algorithms. Additionally, our algorithm accurately simulates the decay and recovery of opinions over time, underscoring LLMs potential to revolutionize the understanding of social media dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03112v1">Plug-and-Play AMC: Context Is King in Training-Free, Open-Set Modulation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Automatic Modulation Classification (AMC) is critical for efficient spectrum management and robust wireless communications. However, AMC remains challenging due to the complex interplay of signal interference and noise. In this work, we propose an innovative framework that integrates traditional signal processing techniques with Large-Language Models (LLMs) to address AMC. Our approach leverages higher-order statistics and cumulant estimation to convert quantitative signal features into structured natural language prompts. By incorporating exemplar contexts into these prompts, our method exploits the LLM's inherent familiarity with classical signal processing, enabling effective one-shot classification without additional training or preprocessing (e.g., denoising). Experimental evaluations on synthetically generated datasets, spanning both noiseless and noisy conditions, demonstrate that our framework achieves competitive performance across diverse modulation schemes and Signal-to-Noise Ratios (SNRs). Moreover, our approach paves the way for robust foundation models in wireless communications across varying channel conditions, significantly reducing the expense associated with developing channel-specific models. This work lays the foundation for scalable, interpretable, and versatile signal classification systems in next-generation wireless networks. The source code is available at https://github.com/RU-SIT/context-is-king
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03100v1">Towards a standardized methodology and dataset for evaluating LLM-based digital forensic timeline analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have seen widespread adoption in many domains including digital forensics. While prior research has largely centered on case studies and examples demonstrating how LLMs can assist forensic investigations, deeper explorations remain limited, i.e., a standardized approach for precise performance evaluations is lacking. Inspired by the NIST Computer Forensic Tool Testing Program, this paper proposes a standardized methodology to quantitatively evaluate the application of LLMs for digital forensic tasks, specifically in timeline analysis. The paper describes the components of the methodology, including the dataset, timeline generation, and ground truth development. Additionally, the paper recommends using BLEU and ROUGE metrics for the quantitative evaluation of LLMs through case studies or tasks involving timeline analysis. Experimental results using ChatGPT demonstrate that the proposed methodology can effectively evaluate LLM-based forensic timeline analysis. Finally, we discuss the limitations of applying LLMs to forensic timeline analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03096v1">Assessing and Enhancing the Robustness of LLM-based Multi-Agent Systems Through Chaos Engineering</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      This study explores the application of chaos engineering to enhance the robustness of Large Language Model-Based Multi-Agent Systems (LLM-MAS) in production-like environments under real-world conditions. LLM-MAS can potentially improve a wide range of tasks, from answering questions and generating content to automating customer support and improving decision-making processes. However, LLM-MAS in production or preproduction environments can be vulnerable to emergent errors or disruptions, such as hallucinations, agent failures, and agent communication failures. This study proposes a chaos engineering framework to proactively identify such vulnerabilities in LLM-MAS, assess and build resilience against them, and ensure reliable performance in critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04249v3">DiffSpec: Differential Testing with LLMs using Natural Language Specifications and Code Artifacts</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Differential testing can be an effective way to find bugs in software systems with multiple implementations that conform to the same specification, like compilers, network protocol parsers, or language runtimes. Specifications for such systems are often standardized in natural language documents, like Instruction Set Architecture (ISA) specifications or IETF RFC's. Large Language Models (LLMs) have demonstrated potential in both generating tests and handling large volumes of natural language text, making them well-suited for analyzing artifacts like specification documents, bug reports, and code implementations. In this work, we leverage natural language and code artifacts to guide LLMs to generate targeted tests that highlight meaningful behavioral differences between implementations, including those corresponding to bugs. We introduce DiffSpec, a framework for generating differential tests with LLMs using prompt chaining. We demonstrate DiffSpec's efficacy on two different (extensively tested) systems, eBPF runtimes and Wasm validators. Using DiffSpec, we generated 1901 differentiating tests, uncovering at least four distinct and confirmed bugs in eBPF, including a kernel memory leak, inconsistent behavior in jump instructions, undefined behavior when using the stack pointer, and tests with infinite loops that hang the verifier in ebpf-for-windows. We also found 299 differentiating tests in Wasm validators pointing to two confirmed and fixed bugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11863v2">Context-aware LLM-based Safe Control Against Latent Risks</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Autonomous control systems face significant challenges in performing complex tasks in the presence of latent risks. To address this, we propose an integrated framework that combines Large Language Models (LLMs), numerical optimization, and optimization-based control to facilitate efficient subtask learning while ensuring safety against latent risks. The framework decomposes complex tasks into a sequence of context-aware subtasks that account for latent risks. These subtasks and their parameters are then refined through a multi-time-scale process: high-layer multi-turn in-context learning, mid-layer LLM Chain-of-Thought reasoning and numerical optimization, and low-layer model predictive control. The framework iteratively improves decisions by leveraging qualitative feedback and optimized trajectory data from lower-layer optimization processes and a physics simulator. We validate the proposed framework through simulated case studies involving robot arm and autonomous vehicle scenarios. The experiments demonstrate that the proposed framework can mediate actions based on the context and latent risks and learn complex behaviors efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04021v1">Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) is expensive, especially for providers hosting many models, making cost reduction essential. The unique workload patterns of serving multiple LLMs (i.e., multi-LLM serving) create new opportunities and challenges for this task. The long-tail popularity of models and their long idle periods present opportunities to improve utilization through GPU sharing. However, existing GPU sharing systems lack the ability to adjust their resource allocation and sharing policies at runtime, making them ineffective at meeting latency service-level objectives (SLOs) under rapidly fluctuating workloads. This paper presents Prism, a multi-LLM serving system that unleashes the full potential of GPU sharing to achieve both cost efficiency and SLO attainment. At its core, Prism tackles a key limitation of existing systems$\unicode{x2014}$the lack of $\textit{cross-model memory coordination}$, which is essential for flexibly sharing GPU memory across models under dynamic workloads. Prism achieves this with two key designs. First, it supports on-demand memory allocation by dynamically mapping physical to virtual memory pages, allowing flexible memory redistribution among models that space- and time-share a GPU. Second, it improves memory efficiency through a two-level scheduling policy that dynamically adjusts sharing strategies based on models' runtime demands. Evaluations on real-world traces show that Prism achieves more than $2\times$ cost savings and $3.3\times$ SLO attainment compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02044v3">Towards Universal and Black-Box Query-Response Only Attack on LLMs with QROA</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Models (LLMs) has exposed critical security and ethical vulnerabilities, particularly their susceptibility to adversarial manipulations. This paper introduces QROA, a novel black-box jailbreak method designed to identify adversarial suffixes that can bypass LLM alignment safeguards when appended to a malicious instruction. Unlike existing suffix-based jailbreak approaches, QROA does not require access to the model's logit or any other internal information. It also eliminates reliance on human-crafted templates, operating solely through the standard query-response interface of LLMs. By framing the attack as an optimization bandit problem, QROA employs a surrogate model and token level optimization to efficiently explore suffix variations. Furthermore, we propose QROA-UNV, an extension that identifies universal adversarial suffixes for individual models, enabling one-query jailbreaks across a wide range of instructions. Testing on multiple models demonstrates Attack Success Rate (ASR) greater than 80\%. These findings highlight critical vulnerabilities, emphasize the need for advanced defenses, and contribute to the development of more robust safety evaluations for secure AI deployment. The code is made public on the following link: https://github.com/qroa/QROA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03973v1">Divide, Optimize, Merge: Fine-Grained LLM Agent Optimization at Scale</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      LLM-based optimization has shown remarkable potential in enhancing agentic systems. However, the conventional approach of prompting LLM optimizer with the whole training trajectories on training dataset in a single pass becomes untenable as datasets grow, leading to context window overflow and degraded pattern recognition. To address these challenges, we propose Fine-Grained Optimization (FGO), a scalable framework that divides large optimization tasks into manageable subsets, performs targeted optimizations, and systematically combines optimized components through progressive merging. Evaluation across ALFWorld, LogisticsQA, and GAIA benchmarks demonstrate that FGO outperforms existing approaches by 1.6-8.6% while reducing average prompt token consumption by 56.3%. Our framework provides a practical solution for scaling up LLM-based optimization of increasingly sophisticated agent systems. Further analysis demonstrates that FGO achieves the most consistent performance gain in all training dataset sizes, showcasing its scalability and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03961v1">The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 16 pages, 8 figures. Code available at https://github.com/storyagents25/story-agents
    </div>
    <details class="paper-abstract">
      According to Yuval Noah Harari, large-scale human cooperation is driven by shared narratives that encode common beliefs and values. This study explores whether such narratives can similarly nudge LLM agents toward collaboration. We use a finitely repeated public goods game in which LLM agents choose either cooperative or egoistic spending strategies. We prime agents with stories highlighting teamwork to different degrees and test how this influences negotiation outcomes. Our experiments explore four questions:(1) How do narratives influence negotiation behavior? (2) What differs when agents share the same story versus different ones? (3) What happens when the agent numbers grow? (4) Are agents resilient against self-serving negotiators? We find that story-based priming significantly affects negotiation strategies and success rates. Common stories improve collaboration, benefiting each agent. By contrast, priming agents with different stories reverses this effect, and those agents primed toward self-interest prevail. We hypothesize that these results carry implications for multi-agent system design and AI alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04654v1">A Comparative Analysis of Ethical and Safety Gaps in LLMs using Relative Danger Coefficient</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) and Large Language Models (LLMs) have rapidly evolved in recent years, showcasing remarkable capabilities in natural language understanding and generation. However, these advancements also raise critical ethical questions regarding safety, potential misuse, discrimination and overall societal impact. This article provides a comparative analysis of the ethical performance of various AI models, including the brand new DeepSeek-V3(R1 with reasoning and without), various GPT variants (4o, 3.5 Turbo, 4 Turbo, o1/o3 mini) and Gemini (1.5 flash, 2.0 flash and 2.0 flash exp) and highlights the need for robust human oversight, especially in situations with high stakes. Furthermore, we present a new metric for calculating harm in LLMs called Relative Danger Coefficient (RDC).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03733v1">WebGen-Bench: Evaluating LLMs on Generating Interactive and Functional Websites from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      LLM-based agents have demonstrated great potential in generating and managing code within complex codebases. In this paper, we introduce WebGen-Bench, a novel benchmark designed to measure an LLM-based agent's ability to create multi-file website codebases from scratch. It contains diverse instructions for website generation, created through the combined efforts of human annotators and GPT-4o. These instructions span three major categories and thirteen minor categories, encompassing nearly all important types of web applications. To assess the quality of the generated websites, we use GPT-4o to generate test cases targeting each functionality described in the instructions, and then manually filter, adjust, and organize them to ensure accuracy, resulting in 647 test cases. Each test case specifies an operation to be performed on the website and the expected result after the operation. To automate testing and improve reproducibility, we employ a powerful web-navigation agent to execute tests on the generated websites and determine whether the observed responses align with the expected results. We evaluate three high-performance code-agent frameworks, Bolt.diy, OpenHands, and Aider, using multiple proprietary and open-source LLMs as engines. The best-performing combination, Bolt.diy powered by DeepSeek-R1, achieves only 27.8\% accuracy on the test cases, highlighting the challenging nature of our benchmark. Additionally, we construct WebGen-Instruct, a training set consisting of 6,667 website-generation instructions. Training Qwen2.5-Coder-32B-Instruct on Bolt.diy trajectories generated from a subset of this training set achieves an accuracy of 38.2\%, surpassing the performance of the best proprietary model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20953v2">Clean & Clear: Feasibility of Safe LLM Clinical Guidance</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Background: Clinical guidelines are central to safe evidence-based medicine in modern healthcare, providing diagnostic criteria, treatment options and monitoring advice for a wide range of illnesses. LLM-empowered chatbots have shown great promise in Healthcare Q&A tasks, offering the potential to provide quick and accurate responses to medical inquiries. Our main objective was the development and preliminary assessment of an LLM-empowered chatbot software capable of reliably answering clinical guideline questions using University College London Hospital (UCLH) clinical guidelines. Methods: We used the open-weight Llama-3.1-8B LLM to extract relevant information from the UCLH guidelines to answer questions. Our approach highlights the safety and reliability of referencing information over its interpretation and response generation. Seven doctors from the ward assessed the chatbot's performance by comparing its answers to the gold standard. Results: Our chatbot demonstrates promising performance in terms of relevance, with ~73% of its responses rated as very relevant, showcasing a strong understanding of the clinical context. Importantly, our chatbot achieves a recall of 1.00 for extracted guideline lines, substantially minimising the risk of missing critical information. Approximately 78% of responses were rated satisfactory in terms of completeness. A small portion (~14.5%) contained minor unnecessary information, indicating occasional lapses in precision. The chatbot' showed high efficiency, with an average completion time of 10 seconds, compared to 30 seconds for human respondents. Evaluation of clinical reasoning showed that 72% of the chatbot's responses were without flaws. Our chatbot demonstrates significant potential to speed up and improve the process of accessing locally relevant clinical information for healthcare professionals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03678v1">Graph Drawing for LLMs: An Empirical Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Our work contributes to the fast-growing literature on the use of Large Language Models (LLMs) to perform graph-related tasks. In particular, we focus on usage scenarios that rely on the visual modality, feeding the model with a drawing of the graph under analysis. We investigate how the model's performance is affected by the chosen layout paradigm, the aesthetics of the drawing, and the prompting technique used for the queries. We formulate three corresponding research questions and present the results of a thorough experimental analysis. Our findings reveal that choosing the right layout paradigm and optimizing the readability of the input drawing from a human perspective can significantly improve the performance of the model on the given task. Moreover, selecting the most effective prompting technique is a challenging yet crucial task for achieving optimal performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15279v2">Don't Mesh with Me: Generating Constructive Solid Geometry Instead of Meshes by Fine-Tuning a Code-Generation LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 Accepted to the AI for Content Creation Workshop at CVPR 2025
    </div>
    <details class="paper-abstract">
      While recent advancements in machine learning, such as LLMs, are revolutionizing software development and creative industries, they have had minimal impact on engineers designing mechanical parts, which remains largely a manual process. Existing approaches to generating 3D geometry most commonly use meshes as a 3D representation. While meshes are suitable for assets in video games or animations, they lack sufficient precision and adaptability for mechanical engineering purposes. This paper introduces a novel approach for the generation of 3D geometry that generates surface-based Constructive Solid Geometry (CSG) by leveraging a code-generation LLM. First, we create a dataset of 3D mechanical parts represented as code scripts by converting Boundary Representation geometry (BREP) into CSG-based Python scripts. Second, we create annotations in natural language using GPT-4. The resulting dataset is used to fine-tune a code-generation LLM. The fine-tuned LLM can complete geometries based on positional input and natural language in a plausible way, demonstrating geometric understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18991v2">HAIR: Hardness-Aware Inverse Reinforcement Learning with Introspective Reasoning for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 The three authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      The alignment of large language models (LLMs) with human values remains critical yet hindered by four key challenges: (1) scarcity of balanced safety datasets, (2) alignment tax, (3) vulnerability to jailbreak attacks due to shallow alignment, and (4) inability to dynamically adapt rewards according to task difficulty. To address these limitations, we introduce HAIR (Hardness-Aware Inverse Reinforcement Learning with Introspective Reasoning), a novel alignment approach inspired by shadow models in membership inference attacks. Our approach consists of two main components: (1) construction of a balanced safety Chain-of-Draft (CoD) dataset for seven harmful categories using structured prompts that leverage the introspective reasoning capabilities of LLMs; and (2) training of category-specific reward models with Group Relative Policy Optimization (GRPO), dynamically tuning optimization to task difficulty at both the data and model levels. Comprehensive experiments across four harmlessness and four usefulness benchmarks demonstrate that HAIR achieves state-of-the-art performance, outperforming all baseline methods in safety while maintaining high levels of usefulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03531v1">Faster MoE LLM Inference for Extremely Large Models</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Sparse Mixture of Experts (MoE) large language models (LLMs) are gradually becoming the mainstream approach for ultra-large-scale models. Existing optimization efforts for MoE models have focused primarily on coarse-grained MoE architectures. With the emergence of DeepSeek Models, fine-grained MoE models are gaining popularity, yet research on them remains limited. Therefore, we want to discuss the efficiency dynamic under different service loads. Additionally, fine-grained models allow deployers to reduce the number of routed experts, both activated counts and total counts, raising the question of how this reduction affects the trade-off between MoE efficiency and performance. Our findings indicate that while deploying MoE models presents greater challenges, it also offers significant optimization opportunities. Reducing the number of activated experts can lead to substantial efficiency improvements in certain scenarios, with only minor performance degradation. Reducing the total number of experts provides limited efficiency gains but results in severe performance degradation. Our method can increase throughput by at least 10\% without any performance degradation. Overall, we conclude that MoE inference optimization remains an area with substantial potential for exploration and improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03475v1">am-ELO: A Stable Framework for Arena-based LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 ICML2025 Accepted
    </div>
    <details class="paper-abstract">
      Arena-based evaluation is a fundamental yet significant evaluation paradigm for modern AI models, especially large language models (LLMs). Existing framework based on ELO rating system suffers from the inevitable instability problem due to ranking inconsistency and the lack of attention to the varying abilities of annotators. In this paper, we introduce a novel stable arena framework to address these issues by enhancing the ELO Rating System. Specifically, we replace the iterative update method with a Maximum Likelihood Estimation (MLE) approach, m-ELO, and provide theoretical proof of the consistency and stability of the MLE approach for model ranking. Additionally, we proposed the am-ELO, which modify the Elo Rating's probability function to incorporate annotator abilities, enabling the simultaneous estimation of model scores and annotator reliability. Experiments demonstrate that this method ensures stability, proving that this framework offers a more robust, accurate, and stable evaluation method for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18902v2">LLMs for Extremely Low-Resource Finno-Ugric Languages</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      The advancement of large language models (LLMs) has predominantly focused on high-resource languages, leaving low-resource languages, such as those in the Finno-Ugric family, significantly underrepresented. This paper addresses this gap by focusing on V\~oro, Livonian, and Komi. We cover almost the entire cycle of LLM creation, from data collection to instruction tuning and evaluation. Our contributions include developing multilingual base and instruction-tuned models; creating evaluation benchmarks, including the smugri-MT-bench multi-turn conversational benchmark; and conducting human evaluation. We intend for this work to promote linguistic diversity, ensuring that lesser-resourced languages can benefit from advancements in NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02722v1">Enhancing LLMs' Clinical Reasoning with Real-World Data from a Nationwide Sepsis Registry</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have demonstrated impressive reasoning capabilities across general domains, their effectiveness in real-world clinical practice remains limited. This is likely due to their insufficient exposure to real-world clinical data during training, as such data is typically not included due to privacy concerns. To address this, we propose enhancing the clinical reasoning capabilities of LLMs by leveraging real-world clinical data. We constructed reasoning-intensive questions from a nationwide sepsis registry and fine-tuned Phi-4 on these questions using reinforcement learning, resulting in C-Reason. C-Reason exhibited strong clinical reasoning capabilities on the in-domain test set, as evidenced by both quantitative metrics and expert evaluations. Furthermore, its enhanced reasoning capabilities generalized to a sepsis dataset involving different tasks and patient cohorts, an open-ended consultations on antibiotics use task, and other diseases. Future research should focus on training LLMs with large-scale, multi-disease clinical datasets to develop more powerful, general-purpose clinical reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02699v1">Exploring LLM-Powered Role and Action-Switching Pedagogical Agents for History Education in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 14 pages excluding reference and appendix. Accepted at ACM CHI 2025. https://dl.acm.org/doi/10.1145/3706598.3713109
    </div>
    <details class="paper-abstract">
      Multi-role pedagogical agents can create engaging and immersive learning experiences, helping learners better understand knowledge in history learning. However, existing pedagogical agents often struggle with multi-role interactions due to complex controls, limited feedback forms, and difficulty dynamically adapting to user inputs. In this study, we developed a VR prototype with LLM-powered adaptive role-switching and action-switching pedagogical agents to help users learn about the history of the Pavilion of Prince Teng. A 2 x 2 between-subjects study was conducted with 84 participants to assess how adaptive role-switching and action-switching affect participants' learning outcomes and experiences. The results suggest that adaptive role-switching enhances participants' perception of the pedagogical agent's trustworthiness and expertise but may lead to inconsistent learning experiences. Adaptive action-switching increases participants' perceived social presence, expertise, and humanness. The study did not uncover any effects of role-switching and action-switching on usability, learning motivation, and cognitive load. Based on the findings, we proposed five design implications for incorporating adaptive role-switching and action-switching into future VR history education tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18881v2">Letters from Future Self: Augmenting the Letter-Exchange Exercise with LLM-based Agents to Enhance Young Adults' Career Exploration</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 21 pages, 9 figures, Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems (Best Paper Award, Top 1%)
    </div>
    <details class="paper-abstract">
      Young adults often encounter challenges in career exploration. Self-guided interventions, such as the letter-exchange exercise, where participants envision and adopt the perspective of their future selves by exchanging letters with their envisioned future selves, can support career development. However, the broader adoption of such interventions may be limited without structured guidance. To address this, we integrated Large Language Model (LLM)-based agents that simulate participants' future selves into the letter-exchange exercise and evaluated their effectiveness. A one-week experiment (N=36) compared three conditions: (1) participants manually writing replies to themselves from the perspective of their future selves (baseline), (2) future-self agents generating letters to participants, and (3) future-self agents engaging in chat conversations with participants. Results indicated that exchanging letters with future-self agents enhanced participants' engagement during the exercise, while overall benefits of the intervention on future orientation, career self-concept, and psychological support remained comparable across conditions. We discuss design implications for AI-augmented interventions for supporting young adults' career exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02693v1">Predicting Movie Hits Before They Happen with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted at ACM UMAP 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Addressing the cold-start issue in content recommendation remains a critical ongoing challenge. In this work, we focus on tackling the cold-start problem for movies on a large entertainment platform. Our primary goal is to forecast the popularity of cold-start movies using Large Language Models (LLMs) leveraging movie metadata. This method could be integrated into retrieval systems within the personalization pipeline or could be adopted as a tool for editorial teams to ensure fair promotion of potentially overlooked movies that may be missed by traditional or algorithmic solutions. Our study validates the effectiveness of this approach compared to established baselines and those we developed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02666v1">A Survey on Progress in LLM Alignment from the Perspective of Reward Design</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The alignment of large language models (LLMs) with human values and intentions represents a core challenge in current AI research, where reward mechanism design has become a critical factor in shaping model behavior. This study conducts a comprehensive investigation of reward mechanisms in LLM alignment through a systematic theoretical framework, categorizing their development into three key phases: (1) feedback (diagnosis), (2) reward design (prescription), and (3) optimization (treatment). Through a four-dimensional analysis encompassing construction basis, format, expression, and granularity, this research establishes a systematic classification framework that reveals evolutionary trends in reward modeling. The field of LLM alignment faces several persistent challenges, while recent advances in reward design are driving significant paradigm shifts. Notable developments include the transition from reinforcement learning-based frameworks to novel optimization paradigms, as well as enhanced capabilities to address complex alignment scenarios involving multimodal integration and concurrent task coordination. Finally, this survey outlines promising future research directions for LLM alignment through innovative reward design strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02665v1">A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      This survey explores recent advancements in reasoning large language models (LLMs) designed to mimic "slow thinking" - a reasoning process inspired by human cognition, as described in Kahneman's Thinking, Fast and Slow. These models, like OpenAI's o1, focus on scaling computational resources dynamically during complex tasks, such as math reasoning, visual reasoning, medical diagnosis, and multi-agent debates. We present the development of reasoning LLMs and list their key technologies. By synthesizing over 100 studies, it charts a path toward LLMs that combine human-like deep thinking with scalable efficiency for reasoning. The review breaks down methods into three categories: (1) test-time scaling dynamically adjusts computation based on task complexity via search and sampling, dynamic verification; (2) reinforced learning refines decision-making through iterative improvement leveraging policy networks, reward models, and self-evolution strategies; and (3) slow-thinking frameworks (e.g., long CoT, hierarchical processes) that structure problem-solving with manageable steps. The survey highlights the challenges and further directions of this domain. Understanding and advancing the reasoning abilities of LLMs is crucial for unlocking their full potential in real-world applications, from scientific discovery to decision support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00711v2">GraphMaster: Automated Graph Synthesis via LLM Agents in Data-Limited Environments</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      The era of foundation models has revolutionized AI research, yet Graph Foundation Models (GFMs) remain constrained by the scarcity of large-scale graph corpora. Traditional graph data synthesis techniques primarily focus on simplistic structural operations, lacking the capacity to generate semantically rich nodes with meaningful textual attributes: a critical limitation for real-world applications. While large language models (LLMs) demonstrate exceptional text generation capabilities, their direct application to graph synthesis is impeded by context window limitations, hallucination phenomena, and structural consistency challenges. To address these issues, we introduce GraphMaster, the first multi-agent framework specifically designed for graph data synthesis in data-limited environments. GraphMaster orchestrates four specialized LLM agents (Manager, Perception, Enhancement, and Evaluation) that collaboratively optimize the synthesis process through iterative refinement, ensuring both semantic coherence and structural integrity. To rigorously evaluate our approach, we create new data-limited "Sub" variants of six standard graph benchmarks, specifically designed to test synthesis capabilities under realistic constraints. Additionally, we develop a novel interpretability assessment framework that combines human evaluation with a principled Grassmannian manifold-based analysis, providing both qualitative and quantitative measures of semantic coherence. Experimental results demonstrate that GraphMaster significantly outperforms traditional synthesis methods across multiple datasets, establishing a strong foundation for advancing GFMs in data-scarce environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02625v1">LLaMA-Omni2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Preprint. Project: https://github.com/ictnlp/LLaMA-Omni2
    </div>
    <details class="paper-abstract">
      Real-time, intelligent, and natural speech interaction is an essential part of the next-generation human-computer interaction. Recent advancements have showcased the potential of building intelligent spoken chatbots based on large language models (LLMs). In this paper, we introduce LLaMA-Omni 2, a series of speech language models (SpeechLMs) ranging from 0.5B to 14B parameters, capable of achieving high-quality real-time speech interaction. LLaMA-Omni 2 is built upon the Qwen2.5 series models, integrating a speech encoder and an autoregressive streaming speech decoder. Despite being trained on only 200K multi-turn speech dialogue samples, LLaMA-Omni 2 demonstrates strong performance on several spoken question answering and speech instruction following benchmarks, surpassing previous state-of-the-art SpeechLMs like GLM-4-Voice, which was trained on millions of hours of speech data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16892v2">Applying LLMs to Active Learning: Towards Cost-Efficient Cross-Task Text Classification without Manually Labeled Data</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Machine learning-based classifiers have been used for text classification, such as sentiment analysis, news classification, and toxic comment classification. However, supervised machine learning models often require large amounts of labeled data for training, and manual annotation is both labor-intensive and requires domain-specific knowledge, leading to relatively high annotation costs. To address this issue, we propose an approach that integrates large language models (LLMs) into an active learning framework, achieving high cross-task text classification performance without the need for any manually labeled data. Furthermore, compared to directly applying GPT for classification tasks, our approach retains over 93% of its classification performance while requiring only approximately 6% of the computational time and monetary cost, effectively balancing performance and resource efficiency. These findings provide new insights into the efficient utilization of LLMs and active learning algorithms in text classification tasks, paving the way for their broader application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02583v1">Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted by IJCAI 2025 Survey Track
    </div>
    <details class="paper-abstract">
      The proliferation of edge devices has generated an unprecedented volume of time series data across different domains, motivating various well-customized methods. Recently, Large Language Models (LLMs) have emerged as a new paradigm for time series analytics by leveraging the shared sequential nature of textual data and time series. However, a fundamental cross-modality gap between time series and LLMs exists, as LLMs are pre-trained on textual corpora and are not inherently optimized for time series. Many recent proposals are designed to address this issue. In this survey, we provide an up-to-date overview of LLMs-based cross-modality modeling for time series analytics. We first introduce a taxonomy that classifies existing approaches into four groups based on the type of textual data employed for time series modeling. We then summarize key cross-modality strategies, e.g., alignment and fusion, and discuss their applications across a range of downstream tasks. Furthermore, we conduct experiments on multimodal datasets from different application domains to investigate effective combinations of textual data and cross-modality strategies for enhancing time series analytics. Finally, we suggest several promising directions for future research. This survey is designed for a range of professionals, researchers, and practitioners interested in LLM-based time series modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02579v1">EMORL: Ensemble Multi-Objective Reinforcement Learning for Efficient and Flexible LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 13 pages, 9 figures, submitted to SIGDIAL 2025 conference
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) for large language model (LLM) fine-tuning show promise in addressing multi-objective tasks but still face significant challenges, including complex objective balancing, low training efficiency, poor scalability, and limited explainability. Leveraging ensemble learning principles, we introduce an Ensemble Multi-Objective RL (EMORL) framework that fine-tunes multiple models with individual objectives while optimizing their aggregation after the training to improve efficiency and flexibility. Our method is the first to aggregate the last hidden states of individual models, incorporating contextual information from multiple objectives. This approach is supported by a hierarchical grid search algorithm that identifies optimal weighted combinations. We evaluate EMORL on counselor reflection generation tasks, using text-scoring LLMs to evaluate the generations and provide rewards during RL fine-tuning. Through comprehensive experiments on the PAIR and Psych8k datasets, we demonstrate the advantages of EMORL against existing baselines: significantly lower and more stable training consumption ($17,529\pm 1,650$ data points and $6,573\pm 147.43$ seconds), improved scalability and explainability, and comparable performance across multiple objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02502v1">Unveiling the Landscape of LLM Deployment in the Wild: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Background: Large language models (LLMs) are increasingly deployed via open-source and commercial frameworks, enabling individuals and organizations to self-host advanced AI capabilities. However, insecure defaults and misconfigurations often expose LLM services to the public Internet, posing significant security and system engineering risks. Aims: This study aims to unveil the current landscape of public-facing LLM deployments in the wild through a large-scale empirical study, focusing on service prevalence, exposure characteristics, systemic vulnerabilities, and associated risks. Method: We conducted an Internet-wide measurement to identify public-facing LLM deployments across 15 frameworks, discovering 320,102 services. We extracted 158 unique API endpoints, grouped into 12 functional categories based on capabilities and security risks. We further analyzed configurations, authentication practices, and geographic distributions, revealing deployment trends and systemic issues in real-world LLM system engineering. Results: Our study shows that public LLM deployments are rapidly growing but often insecure. Among all endpoints, we observe widespread use of insecure protocols, poor TLS configurations, and unauthenticated access to critical operations. Security risks, including model disclosure, system leakage, and unauthorized access, are pervasive, highlighting the need for secure-by-default frameworks and stronger deployment practices. Conclusions: Public-facing LLM deployments suffer from widespread security and configuration flaws, exposing services to misuse, model theft, resource hijacking, and remote exploitation. Strengthening default security, deployment practices, and operational standards is critical for the growing self-hosted LLM ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02456v1">Colombian Waitresses y Jueces canadienses: Gender and Country Biases in Occupation Recommendations from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      One of the goals of fairness research in NLP is to measure and mitigate stereotypical biases that are propagated by NLP systems. However, such work tends to focus on single axes of bias (most often gender) and the English language. Addressing these limitations, we contribute the first study of multilingual intersecting country and gender biases, with a focus on occupation recommendations generated by large language models. We construct a benchmark of prompts in English, Spanish and German, where we systematically vary country and gender, using 25 countries and four pronoun sets. Then, we evaluate a suite of 5 Llama-based models on this benchmark, finding that LLMs encode significant gender and country biases. Notably, we find that even when models show parity for gender or country individually, intersectional occupational biases based on both country and gender persist. We also show that the prompting language significantly affects bias, and instruction-tuned models consistently demonstrate the lowest and most stable levels of bias. Our findings highlight the need for fairness researchers to use intersectional and multilingual lenses in their work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02428v1">Can LLM-Simulated Practice and Feedback Upskill Human Counselors? A Randomized Study with 90+ Novice Counselors</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 main paper is 11 pages, with methods it is 18 pages, with appendix and references it is 33 pages
    </div>
    <details class="paper-abstract">
      Training more counselors, from clinical students to peer supporters, can help meet the demand for accessible mental health support; however, current training approaches remain resource-intensive and difficult to scale effectively. Large Language Models (LLMs) offer promising solutions for growing counseling skills training through simulated practice and automated feedback. Despite successes in aligning LLMs with expert-counselor annotations, we do not know whether LLM-based counseling training tools -- such as AI patients that simulate real-world challenges and generative AI feedback with suggested alternatives and rationales -- actually lead to improvements in novice counselor skill development. We develop CARE, an LLM-simulated practice and feedback system, and randomize 94 novice counselors to practice using an AI patient, either alone or with AI feedback, measuring changes in their behavioral performance, self-assessments, and qualitative learning takeaways. Our results show the practice-and-feedback group improved in their use of reflections and questions (d=0.32-0.39, p$<$0.05). In contrast, the group that practiced with an AI patient alone did not show improvements, and in the case of empathy, actually had worse uses across time (d=$-$0.52, p=0.001) and when compared against the practice-and-feedback group (d=0.72, p=0.001). Participants' qualitative self-reflections revealed key differences: the practice-and-feedback group adopted a client-centered approach involving listening to and validating feelings, while the practice-alone group remained solution-oriented but delayed offering suggestions until gathering more information. Overall, these results suggest that LLM-based training systems can promote effective skill development, but that combining both simulated practice and structured feedback is critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02418v1">SymbioticRAG: Enhancing Document Intelligence Through Human-LLM Symbiotic Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      We present \textbf{SymbioticRAG}, a novel framework that fundamentally reimagines Retrieval-Augmented Generation~(RAG) systems by establishing a bidirectional learning relationship between humans and machines. Our approach addresses two critical challenges in current RAG systems: the inherently human-centered nature of relevance determination and users' progression from "unconscious incompetence" in query formulation. SymbioticRAG introduces a two-tier solution where Level 1 enables direct human curation of retrieved content through interactive source document exploration, while Level 2 aims to build personalized retrieval models based on captured user interactions. We implement Level 1 through three key components: (1)~a comprehensive document processing pipeline with specialized models for layout detection, OCR, and extraction of tables, formulas, and figures; (2)~an extensible retriever module supporting multiple retrieval strategies; and (3)~an interactive interface that facilitates both user engagement and interaction data logging. We experiment Level 2 implementation via a retriever strategy incorporated LLM summarized user intention from user interaction logs. To maintain high-quality data preparation, we develop a human-on-the-loop validation interface that improves pipeline output while advancing research in specialized extraction tasks. Evaluation across three scenarios (literature review, geological exploration, and education) demonstrates significant improvements in retrieval relevance and user satisfaction compared to traditional RAG approaches. To facilitate broader research and further advancement of SymbioticRAG Level 2 implementation, we will make our system openly accessible to the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15210v2">Integrating Symbolic Execution into the Fine-Tuning of Code-Generating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Code-generating Large Language Models (LLMs) have become essential tools in modern software development, enhancing productivity and accelerating development. This paper aims to investigate the fine-tuning of code-generating LLMs using Reinforcement Learning and Direct Preference Optimization, further improving their performance. To achieve this, we enhance the training data for the reward model with the help of symbolic execution techniques, ensuring more comprehensive and objective data. With symbolic execution, we create a custom dataset that better captures the nuances in code evaluation. Our reward models, fine-tuned on this dataset, demonstrate significant improvements over the baseline, CodeRL, in estimating the quality of generated code. Our code-generating LLMs, trained with the help of reward model feedback, achieve similar results compared to the CodeRL benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02376v1">LAMeD: LLM-generated Annotations for Memory Leak Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Static analysis tools are widely used to detect software bugs and vulnerabilities but often struggle with scalability and efficiency in complex codebases. Traditional approaches rely on manually crafted annotations -- labeling functions as sources or sinks -- to track data flows, e.g., ensuring that allocated memory is eventually freed, and code analysis tools such as CodeQL, Infer, or Cooddy can use function specifications, but manual annotation is laborious and error-prone, especially for large or third-party libraries. We present LAMeD (LLM-generated Annotations for Memory leak Detection), a novel approach that leverages large language models (LLMs) to automatically generate function-specific annotations. When integrated with analyzers such as Cooddy, LAMeD significantly improves memory leak detection and reduces path explosion. We also suggest directions for extending LAMeD to broader code analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00985v2">Position: Enough of Scaling LLMs! Lets Focus on Downscaling</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      We challenge the dominant focus on neural scaling laws and advocate for a paradigm shift toward downscaling in the development of large language models (LLMs). While scaling laws have provided critical insights into performance improvements through increasing model and dataset size, we emphasize the significant limitations of this approach, particularly in terms of computational inefficiency, environmental impact, and deployment constraints. To address these challenges, we propose a holistic framework for downscaling LLMs that seeks to maintain performance while drastically reducing resource demands. This paper outlines practical strategies for transitioning away from traditional scaling paradigms, advocating for a more sustainable, efficient, and accessible approach to LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00626v2">The Illusion of Role Separation: Hidden Shortcuts in LLM Role Learning (and How to Fix Them)</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) that integrate multiple input roles (e.g., system instructions, user queries, external tool outputs) are increasingly prevalent in practice. Ensuring that the model accurately distinguishes messages from each role -- a concept we call \emph{role separation} -- is crucial for consistent multi-role behavior. Although recent work often targets state-of-the-art prompt injection defenses, it remains unclear whether such methods truly teach LLMs to differentiate roles or merely memorize known triggers. In this paper, we examine \emph{role-separation learning}: the process of teaching LLMs to robustly distinguish system and user tokens. Through a \emph{simple, controlled experimental framework}, we find that fine-tuned models often rely on two proxies for role identification: (1) task type exploitation, and (2) proximity to begin-of-text. Although data augmentation can partially mitigate these shortcuts, it generally leads to iterative patching rather than a deeper fix. To address this, we propose reinforcing \emph{invariant signals} that mark role boundaries by adjusting token-wise cues in the model's input encoding. In particular, manipulating position IDs helps the model learn clearer distinctions and reduces reliance on superficial proxies. By focusing on this mechanism-centered perspective, our work illuminates how LLMs can more reliably maintain consistent multi-role behavior without merely memorizing known prompts or triggers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02324v1">From Course to Skill: Evaluating LLM Performance in Curricular Analytics</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Curricular analytics (CA) -- systematic analysis of curricula data to inform program and course refinement -- becomes an increasingly valuable tool to help institutions align academic offerings with evolving societal and economic demands. Large language models (LLMs) are promising for handling large-scale, unstructured curriculum data, but it remains uncertain how reliably LLMs can perform CA tasks. In this paper, we systematically evaluate four text alignment strategies based on LLMs or traditional NLP methods for skill extraction, a core task in CA. Using a stratified sample of 400 curriculum documents of different types and a human-LLM collaborative evaluation framework, we find that retrieval-augmented generation (RAG) to be the top-performing strategy across all types of curriculum documents, while zero-shot prompting performs worse than traditional NLP methods in most cases. Our findings highlight the promise of LLMs in analyzing brief and abstract curriculum documents, but also reveal that their performance can vary significantly depending on model selection and prompting strategies. This underscores the importance of carefully evaluating the performance of LLM-based strategies before large-scale deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02322v1">HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 arXiv admin note: text overlap with arXiv:2406.14228 by other authors
    </div>
    <details class="paper-abstract">
      Recent advancements have significantly enhanced the performance of large language models (LLMs) in tackling complex reasoning tasks, achieving notable success in domains like mathematical and logical reasoning. However, these methods encounter challenges with complex planning tasks, primarily due to extended reasoning steps, diverse constraints, and the challenge of handling multiple distinct sub-tasks. To address these challenges, we propose HyperTree Planning (HTP), a novel reasoning paradigm that constructs hypertree-structured planning outlines for effective planning. The hypertree structure enables LLMs to engage in hierarchical thinking by flexibly employing the divide-and-conquer strategy, effectively breaking down intricate reasoning steps, accommodating diverse constraints, and managing multiple distinct sub-tasks in a well-organized manner. We further introduce an autonomous planning framework that completes the planning process by iteratively refining and expanding the hypertree-structured planning outlines. Experiments demonstrate the effectiveness of HTP, achieving state-of-the-art accuracy on the TravelPlanner benchmark with Gemini-1.5-Pro, resulting in a 3.6 times performance improvement over o1-preview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02309v1">Optimizing LLMs for Resource-Constrained Environments: A Survey of Model Compression Techniques</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted to IEEE COMPSAC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized many areas of artificial intelligence (AI), but their substantial resource requirements limit their deployment on mobile and edge devices. This survey paper provides a comprehensive overview of techniques for compressing LLMs to enable efficient inference in resource-constrained environments. We examine three primary approaches: Knowledge Distillation, Model Quantization, and Model Pruning. For each technique, we discuss the underlying principles, present different variants, and provide examples of successful applications. We also briefly discuss complementary techniques such as mixture-of-experts and early-exit strategies. Finally, we highlight promising future directions, aiming to provide a valuable resource for both researchers and practitioners seeking to optimize LLMs for edge deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03059v1">Improving Model Alignment Through Collective Intelligence of Open-Source LLMS</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Building helpful and harmless large language models (LLMs) requires effective model alignment approach based on human instructions and feedback, which necessitates high-quality human-labeled data. Constructing such datasets is often expensive and hard to scale, and may face potential limitations on diversity and generalization. To address these challenges, we introduce Mixture of Agents Alignment (MoAA), that leverages the collective strengths of various language models to provide high-quality data for model alignment. By employing MoAA, we enhance both supervised fine-tuning and preference optimization, leading to improved performance compared to using a single model alone to generate alignment data (e.g. using GPT-4o alone). Evaluation results show that our approach can improve win rate of LLaMA-3.1-8B-Instruct from 19.5 to 48.3 on Arena-Hard and from 22.33 to 57.23 on AlpacaEval2, highlighting a promising direction for model alignment through this new scalable and diverse synthetic data recipe. Furthermore, we demonstrate that MoAA enables a self-improvement pipeline, where models finetuned on MoA-generated data surpass their own initial capabilities, providing evidence that our approach can push the frontier of open-source LLMs without reliance on stronger external supervision. Data and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03049v1">34 Examples of LLM Applications in Materials Science and Chemistry: Towards Automation, Assistants, Agents, and Accelerated Scientific Discovery</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.15221
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping many aspects of materials science and chemistry research, enabling advances in molecular property prediction, materials design, scientific automation, knowledge extraction, and more. Recent developments demonstrate that the latest class of models are able to integrate structured and unstructured data, assist in hypothesis generation, and streamline research workflows. To explore the frontier of LLM capabilities across the research lifecycle, we review applications of LLMs through 34 total projects developed during the second annual Large Language Model Hackathon for Applications in Materials Science and Chemistry, a global hybrid event. These projects spanned seven key research areas: (1) molecular and material property prediction, (2) molecular and material design, (3) automation and novel interfaces, (4) scientific communication and education, (5) research data management and automation, (6) hypothesis generation and evaluation, and (7) knowledge extraction and reasoning from the scientific literature. Collectively, these applications illustrate how LLMs serve as versatile predictive models, platforms for rapid prototyping of domain-specific tools, and much more. In particular, improvements in both open source and proprietary LLM performance through the addition of reasoning, additional training data, and new techniques have expanded effectiveness, particularly in low-data environments and interdisciplinary research. As LLMs continue to improve, their integration into scientific workflows presents both new opportunities and new challenges, requiring ongoing exploration, continued refinement, and further research to address reliability, interpretability, and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03030v1">UCSC at SemEval-2025 Task 3: Context, Models and Prompt Optimization for Automated Hallucination Detection in LLM Output</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 6 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Hallucinations pose a significant challenge for large language models when answering knowledge-intensive queries. As LLMs become more widely adopted, it is crucial not only to detect if hallucinations occur but also to pinpoint exactly where in the LLM output they occur. SemEval 2025 Task 3, Mu-SHROOM: Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes, is a recent effort in this direction. This paper describes the UCSC system submission to the shared Mu-SHROOM task. We introduce a framework that first retrieves relevant context, next identifies false content from the answer, and finally maps them back to spans in the LLM output. The process is further enhanced by automatically optimizing prompts. Our system achieves the highest overall performance, ranking #1 in average position across all languages. We release our code and experiment results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14498v3">LLaSA: A Multimodal LLM for Human Activity Analysis Through Wearable and Smartphone Sensors</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Wearables generate rich motion data, yet current systems only classify what happened - failing to support natural questions about why it happened or what it means. We introduce LLaSA (Large Language and Sensor Assistant), a compact 13B model that enables ask-anything, open-ended question answering grounded in raw IMU data. LLaSA supports conversational, context-aware reasoning - explaining the causes of sensor-detected behaviors and answering free-form questions in real-world scenarios. It is tuned for scientific accuracy, coherence, and response reliability. To advance this new task of sensor-based QA, we release three large-scale datasets: SensorCaps, OpenSQA, and Tune-OpenSQA. Together, these resources define a new benchmark for sensor-language models. LLaSA consistently produces interpretable, causal answers and outperforms commercial LLMs across both public and real-world settings. Our code repository and datasets can be found at https://github.com/BASHLab/LLaSA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14307v2">LLM-3D Print: Large Language Models To Monitor and Control 3D Printing</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Industry 4.0 has revolutionized manufacturing by driving digitalization and shifting the paradigm toward additive manufacturing (AM). Fused Deposition Modeling (FDM), a key AM technology, enables the creation of highly customized, cost-effective products with minimal material waste through layer-by-layer extrusion, posing a significant challenge to traditional subtractive methods. However, the susceptibility of material extrusion techniques to errors often requires expert intervention to detect and mitigate defects that can severely compromise product quality. While automated error detection and machine learning models exist, their generalizability across diverse 3D printer setups, firmware, and sensors is limited, and deep learning methods require extensive labeled datasets, hindering scalability and adaptability. To address these challenges, we present a process monitoring and control framework that leverages pre-trained Large Language Models (LLMs) alongside 3D printers to detect and address printing defects. The LLM evaluates print quality by analyzing images captured after each layer or print segment, identifying failure modes and querying the printer for relevant parameters. It then generates and executes a corrective action plan. We validated the effectiveness of the proposed framework in identifying defects by comparing it against a control group of engineers with diverse AM expertise. Our evaluation demonstrated that LLM-based agents not only accurately identify common 3D printing errors, such as inconsistent extrusion, stringing, warping, and layer adhesion, but also effectively determine the parameters causing these failures and autonomously correct them without any need for human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01822v3">Firewalls to Secure Dynamic LLM Agentic Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Future LLM agents are likely to communicate on behalf of users with other entity-representing agents on tasks that entail long-horizon plans with interdependent goals. Current work does not focus on such agentic networks, nor does it address their challenges. Thus, we first identify the required properties of agents' communication, which should be proactive and adaptable. It needs to satisfy 1) privacy: agents should not share more than what is needed for the task, and 2) security: the communication must preserve integrity and maintain utility against selfish entities. We design a use case (travel planning) as a testbed that exemplifies these requirements, and we show examples of how this can go wrong. Next, we propose a practical design, inspired by established network security principles, for constrained LLM agentic networks that balance adaptability, security, and privacy. Our framework automatically constructs and updates task-specific rules from prior simulations to build firewalls. We offer layers of defense to 1) convert free-form input to a task-specific protocol, 2) dynamically abstract users' data to a task-specific degree of permissiveness, and 3) self-correct the agents' trajectory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03019v1">Memorization or Interpolation ? Detecting LLM Memorization through Input Perturbation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) achieve remarkable performance through training on massive datasets, they can exhibit concerning behaviors such as verbatim reproduction of training data rather than true generalization. This memorization phenomenon raises significant concerns about data privacy, intellectual property rights, and the reliability of model evaluations. This paper introduces PEARL, a novel approach for detecting memorization in LLMs. PEARL assesses how sensitive an LLM's performance is to input perturbations, enabling memorization detection without requiring access to the model's internals. We investigate how input perturbations affect the consistency of outputs, enabling us to distinguish between true generalization and memorization. Our findings, following extensive experiments on the Pythia open model, provide a robust framework for identifying when the model simply regurgitates learned information. Applied on the GPT 4o models, the PEARL framework not only identified cases of memorization of classic texts from the Bible or common code from HumanEval but also demonstrated that it can provide supporting evidence that some data, such as from the New York Times news articles, were likely part of the training data of a given model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07963v3">Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature?</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 22 pages, 12 figures, 4 tables, CHIL 2025
    </div>
    <details class="paper-abstract">
      Medical research faces well-documented challenges in translating novel treatments into clinical practice. Publishing incentives encourage researchers to present "positive" findings, even when empirical results are equivocal. Consequently, it is well-documented that authors often spin study results, especially in article abstracts. Such spin can influence clinician interpretation of evidence and may affect patient care decisions. In this study, we ask whether the interpretation of trial results offered by Large Language Models (LLMs) is similarly affected by spin. This is important since LLMs are increasingly being used to trawl through and synthesize published medical evidence. We evaluated 22 LLMs and found that they are across the board more susceptible to spin than humans. They might also propagate spin into their outputs: We find evidence, e.g., that LLMs implicitly incorporate spin into plain language summaries that they generate. We also find, however, that LLMs are generally capable of recognizing spin, and can be prompted in a way to mitigate spin's impact on LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00153v3">Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Probing learned concepts in large language models (LLMs) is crucial for understanding how semantic knowledge is encoded internally. Training linear classifiers on probing tasks is a principle approach to denote the vector of a certain concept in the representation space. However, the single vector identified for a concept varies with both data and training, making it less robust and weakening its effectiveness in real-world applications. To address this challenge, we propose an approach to approximate the subspace representing a specific concept. Built on linear probing classifiers, we extend the concept vectors into Gaussian Concept Subspace (GCS). We demonstrate GCS's effectiveness through measuring its faithfulness and plausibility across multiple LLMs with different sizes and architectures. Additionally, we use representation intervention tasks to showcase its efficacy in real-world applications such as emotion steering. Experimental results indicate that GCS concept vectors have the potential to balance steering performance and maintaining the fluency in natural language generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01122v2">CFBench: A Comprehensive Constraints-Following Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 15 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The adeptness of Large Language Models (LLMs) in comprehending and following natural language instructions is critical for their deployment in sophisticated real-world applications. Existing evaluations mainly focus on fragmented constraints or narrow scenarios, but they overlook the comprehensiveness and authenticity of constraints from the user's perspective. To bridge this gap, we propose CFBench, a large-scale Comprehensive Constraints Following Benchmark for LLMs, featuring 1,000 curated samples that cover more than 200 real-life scenarios and over 50 NLP tasks. CFBench meticulously compiles constraints from real-world instructions and constructs an innovative systematic framework for constraint types, which includes 10 primary categories and over 25 subcategories, and ensures each constraint is seamlessly integrated within the instructions. To make certain that the evaluation of LLM outputs aligns with user perceptions, we propose an advanced methodology that integrates multi-dimensional assessment criteria with requirement prioritization, covering various perspectives of constraints, instructions, and requirement fulfillment. Evaluating current leading LLMs on CFBench reveals substantial room for improvement in constraints following, and we further investigate influencing factors and enhancement strategies. The data and code are publicly available at https://github.com/PKU-Baichuan-MLSystemLab/CFBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02922v1">RetroInfer: A Vector-Storage Approach for Scalable Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      The growing context lengths of large language models (LLMs) pose significant challenges for efficient inference, primarily due to GPU memory and bandwidth constraints. We present RetroInfer, a novel system that reconceptualizes the key-value (KV) cache as a vector storage system which exploits the inherent attention sparsity to accelerate long-context LLM inference. At its core is the wave index, an Attention-aWare VEctor index that enables efficient and accurate retrieval of critical tokens through techniques such as tripartite attention approximation, accuracy-bounded attention estimation, and segmented clustering. Complementing this is the wave buffer, which coordinates KV cache placement and overlaps computation and data transfer across GPU and CPU to sustain high throughput. Unlike prior sparsity-based methods that struggle with token selection and hardware coordination, RetroInfer delivers robust performance without compromising model accuracy. Experiments on long-context benchmarks show up to 4.5X speedup over full attention within GPU memory limits and up to 10.5X over sparse attention baselines when KV cache is extended to CPU memory, all while preserving full-attention-level accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02881v1">Rewriting Pre-Training Data Boosts LLM Performance in Math and Code</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 27pages(including appendix), 10 figures
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04640v1">A Comparative Benchmark of a Moroccan Darija Toxicity Detection Model (Typica.ai) and Major LLM-Based Moderation APIs (OpenAI, Mistral, Anthropic)</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 GitHub repository with reproducibility materials and evaluation notebook available at: https://github.com/assoudi-typica-ai/darija-toxicity-benchmark
    </div>
    <details class="paper-abstract">
      This paper presents a comparative benchmark evaluating the performance of Typica.ai's custom Moroccan Darija toxicity detection model against major LLM-based moderation APIs: OpenAI (omni-moderation-latest), Mistral (mistral-moderation-latest), and Anthropic Claude (claude-3-haiku-20240307). We focus on culturally grounded toxic content, including implicit insults, sarcasm, and culturally specific aggression often overlooked by general-purpose systems. Using a balanced test set derived from the OMCD_Typica.ai_Mix dataset, we report precision, recall, F1-score, and accuracy, offering insights into challenges and opportunities for moderation in underrepresented languages. Our results highlight Typica.ai's superior performance, underlining the importance of culturally adapted models for reliable content moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05494v1">An Automated LLM-based Pipeline for Asset-Level Database Creation to Assess Deforestation Impact</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted to ACL ClimateNLP 2025
    </div>
    <details class="paper-abstract">
      The European Union Deforestation Regulation (EUDR) requires companies to prove their products do not contribute to deforestation, creating a critical demand for precise, asset-level environmental impact data. Current databases lack the necessary detail, relying heavily on broad financial metrics and manual data collection, which limits regulatory compliance and accurate environmental modeling. This study presents an automated, end-to-end data extraction pipeline that uses LLMs to create, clean, and validate structured databases, specifically targeting sectors with a high risk of deforestation. The pipeline introduces Instructional, Role-Based, Zero-Shot Chain-of-Thought (IRZ-CoT) prompting to enhance data extraction accuracy and a Retrieval-Augmented Validation (RAV) process that integrates real-time web searches for improved data reliability. Applied to SEC EDGAR filings in the Mining, Oil & Gas, and Utilities sectors, the pipeline demonstrates significant improvements over traditional zero-shot prompting approaches, particularly in extraction accuracy and validation coverage. This work advances NLP-driven automation for regulatory compliance, CSR (Corporate Social Responsibility), and ESG, with broad sectoral applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02802v1">Generating HomeAssistant Automations Using an LLM-based Chatbot</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      To combat climate change, individuals are encouraged to adopt sustainable habits, in particular, with their household, optimizing their electrical consumption. Conversational agents, such as Smart Home Assistants, hold promise as effective tools for promoting sustainable practices within households. Our research investigated the application of Large Language Models (LLM) in enhancing smart home automation and promoting sustainable household practices, specifically using the HomeAssistant framework. In particular, it highlights the potential of GPT models in generating accurate automation routines. While the LLMs showed proficiency in understanding complex commands and creating valid JSON outputs, challenges such as syntax errors and message malformations were noted, indicating areas for further improvement. Still, despite minimal quantitative differences between "green" and "no green" prompts, qualitative feedback highlighted a positive shift towards sustainability in the routines generated with environmentally focused prompts. Then, an empirical evaluation (N=56) demonstrated that the system was well-received and found engaging by users compared to its traditional rule-based counterpart. Our findings highlight the role of LLMs in advancing smart home technologies and suggest further research to refine these models for broader, real-world applications to support sustainable living.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21239v4">Semantic Volume: Quantifying and Detecting both External and Internal Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across diverse tasks by encoding vast amounts of factual knowledge. However, they are still prone to hallucinations, generating incorrect or misleading information, often accompanied by high uncertainty. Existing methods for hallucination detection primarily focus on quantifying internal uncertainty, which arises from missing or conflicting knowledge within the model. However, hallucinations can also stem from external uncertainty, where ambiguous user queries lead to multiple possible interpretations. In this work, we introduce Semantic Volume, a novel mathematical measure for quantifying both external and internal uncertainty in LLMs. Our approach perturbs queries and responses, embeds them in a semantic space, and computes the determinant of the Gram matrix of the embedding vectors, capturing their dispersion as a measure of uncertainty. Our framework provides a generalizable and unsupervised uncertainty detection method without requiring internal access to LLMs. We conduct extensive experiments on both external and internal uncertainty detection, demonstrating that our Semantic Volume method consistently outperforms existing baselines in both tasks. Additionally, we provide theoretical insights linking our measure to differential entropy, unifying and extending previous sampling-based uncertainty measures such as the semantic entropy. Semantic Volume is shown to be a robust and interpretable approach to improving the reliability of LLMs by systematically detecting uncertainty in both user queries and model responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20834v2">Token-Efficient RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Title updated to "Token-Efficient RL for LLM Reasoning" to better reflect algorithmic focus. Revised abstract, intro, and conclusion. Paper shortened and typos fixed
    </div>
    <details class="paper-abstract">
      We propose reinforcement learning (RL) strategies tailored for reasoning in large language models (LLMs) under strict memory and compute limits, with a particular focus on compatibility with LoRA fine-tuning. Rather than relying on full-sequence updates or separate critic networks, we design critic-free methods that operate on a small, informative subset of output tokens to reduce memory usage and stabilize training. We introduce S-GRPO, a stochastic variant of Group Relative Policy Optimization, and T-SPMO, a token-level prefix matching approach for fine-grained credit assignment. Applied to Qwen2-1.5B, our methods raise accuracy on the SVAMP benchmark from 46% to over 70% and show strong performance on multi-digit multiplication. Surprisingly, full-token GRPO under LoRA fails to improve over the base model, suggesting that selective token-level optimization may act as an implicit regularizer in low-parameter training regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02124v1">GRAIL: Graph Edit Distance and Node Alignment Using LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Graph Edit Distance (GED) is a widely used metric for measuring similarity between two graphs. Computing the optimal GED is NP-hard, leading to the development of various neural and non-neural heuristics. While neural methods have achieved improved approximation quality compared to non-neural approaches, they face significant challenges: (1) They require large amounts of ground truth data, which is itself NP-hard to compute. (2) They operate as black boxes, offering limited interpretability. (3) They lack cross-domain generalization, necessitating expensive retraining for each new dataset. We address these limitations with GRAIL, introducing a paradigm shift in this domain. Instead of training a neural model to predict GED, GRAIL employs a novel combination of large language models (LLMs) and automated prompt tuning to generate a program that is used to compute GED. This shift from predicting GED to generating programs imparts various advantages, including end-to-end interpretability and an autonomous self-evolutionary learning mechanism without ground-truth supervision. Extensive experiments on seven datasets confirm that GRAIL not only surpasses state-of-the-art GED approximation methods in prediction quality but also achieves robust cross-domain generalization across diverse graph distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02123v1">DriveAgent: Multi-Agent Structured Reasoning with LLM and Multimodal Sensor Fusion for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      We introduce DriveAgent, a novel multi-agent autonomous driving framework that leverages large language model (LLM) reasoning combined with multimodal sensor fusion to enhance situational understanding and decision-making. DriveAgent uniquely integrates diverse sensor modalities-including camera, LiDAR, GPS, and IMU-with LLM-driven analytical processes structured across specialized agents. The framework operates through a modular agent-based pipeline comprising four principal modules: (i) a descriptive analysis agent identifying critical sensor data events based on filtered timestamps, (ii) dedicated vehicle-level analysis conducted by LiDAR and vision agents that collaboratively assess vehicle conditions and movements, (iii) environmental reasoning and causal analysis agents explaining contextual changes and their underlying mechanisms, and (iv) an urgency-aware decision-generation agent prioritizing insights and proposing timely maneuvers. This modular design empowers the LLM to effectively coordinate specialized perception and reasoning agents, delivering cohesive, interpretable insights into complex autonomous driving scenarios. Extensive experiments on challenging autonomous driving datasets demonstrate that DriveAgent is achieving superior performance on multiple metrics against baseline methods. These results validate the efficacy of the proposed LLM-driven multi-agent sensor fusion framework, underscoring its potential to substantially enhance the robustness and reliability of autonomous driving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02099v1">MemEngine: A Unified and Modular Library for Developing Advanced Memory of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Just accepted by TheWebConf'25 Resource Track
    </div>
    <details class="paper-abstract">
      Recently, large language model based (LLM-based) agents have been widely applied across various fields. As a critical part, their memory capabilities have captured significant interest from both industrial and academic communities. Despite the proposal of many advanced memory models in recent research, however, there remains a lack of unified implementations under a general framework. To address this issue, we develop a unified and modular library for developing advanced memory models of LLM-based agents, called MemEngine. Based on our framework, we implement abundant memory models from recent research works. Additionally, our library facilitates convenient and extensible memory development, and offers user-friendly and pluggable memory usage. For benefiting our community, we have made our project publicly available at https://github.com/nuster1128/MemEngine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02091v1">LLM-OptiRA: LLM-Driven Optimization of Resource Allocation for Non-Convex Problems in Wireless Communications</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 6 pages,4 figures
    </div>
    <details class="paper-abstract">
      Solving non-convex resource allocation problems poses significant challenges in wireless communication systems, often beyond the capability of traditional optimization techniques. To address this issue, we propose LLM-OptiRA, the first framework that leverages large language models (LLMs) to automatically detect and transform non-convex components into solvable forms, enabling fully automated resolution of non-convex resource allocation problems in wireless communication systems. LLM-OptiRA not only simplifies problem-solving by reducing reliance on expert knowledge, but also integrates error correction and feasibility validation mechanisms to ensure robustness. Experimental results show that LLM-OptiRA achieves an execution rate of 96% and a success rate of 80% on GPT-4, significantly outperforming baseline approaches in complex optimization tasks across diverse scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02076v1">Leveraging LLM Agents and Digital Twins for Fault Handling in Process Plants</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Advances in Automation and Artificial Intelligence continue to enhance the autonomy of process plants in handling various operational scenarios. However, certain tasks, such as fault handling, remain challenging, as they rely heavily on human expertise. This highlights the need for systematic, knowledge-based methods. To address this gap, we propose a methodological framework that integrates Large Language Model (LLM) agents with a Digital Twin environment. The LLM agents continuously interpret system states and initiate control actions, including responses to unexpected faults, with the goal of returning the system to normal operation. In this context, the Digital Twin acts both as a structured repository of plant-specific engineering knowledge for agent prompting and as a simulation platform for the systematic validation and verification of the generated corrective control actions. The evaluation using a mixing module of a process plant demonstrates that the proposed framework is capable not only of autonomously controlling the mixing module, but also of generating effective corrective actions to mitigate a pipe clogging with only a few reprompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02009v1">Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale datasets for Responsible LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become integral to various real-world applications, leveraging massive, web-sourced datasets like Common Crawl, C4, and FineWeb for pretraining. While these datasets provide linguistic data essential for high-quality natural language generation, they often contain harmful content, such as hate speech, misinformation, and biased narratives. Training LLMs on such unfiltered data risks perpetuating toxic behaviors, spreading misinformation, and amplifying societal biases which can undermine trust in LLM-driven applications and raise ethical concerns about their use. This paper presents a large-scale analysis of inappropriate content across these datasets, offering a comprehensive taxonomy that categorizes harmful webpages into Topical and Toxic based on their intent. We also introduce a prompt evaluation dataset, a high-accuracy Topical and Toxic Prompt (TTP), and a transformer-based model (HarmFormer) for content filtering. Additionally, we create a new multi-harm open-ended toxicity benchmark (HAVOC) and provide crucial insights into how models respond to adversarial toxic inputs. Upon publishing, we will also opensource our model signal on the entire C4 dataset. Our work offers insights into ensuring safer LLM pretraining and serves as a resource for Responsible AI (RAI) compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09061v3">CRANE: Reasoning with constrained LLM generation</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Accepted at ICML 2025
    </div>
    <details class="paper-abstract">
      Code generation, symbolic math reasoning, and other tasks require LLMs to produce outputs that are both syntactically and semantically correct. Constrained LLM generation is a promising direction to enforce adherence to formal grammar, but prior works have empirically observed that strict enforcement of formal constraints often diminishes the reasoning capabilities of LLMs. In this work, we first provide a theoretical explanation for why constraining LLM outputs to very restrictive grammars that only allow syntactically valid final answers reduces the reasoning capabilities of the model. Second, we demonstrate that by augmenting the output grammar with carefully designed additional rules, it is always possible to preserve the reasoning capabilities of the LLM while ensuring syntactic and semantic correctness in its outputs. Building on these theoretical insights, we propose a reasoning-augmented constrained decoding algorithm, CRANE, which effectively balances the correctness of constrained generation with the flexibility of unconstrained generation. Experiments on multiple open-source LLMs and benchmarks show that CRANE significantly outperforms both state-of-the-art constrained decoding strategies and standard unconstrained decoding, showing up to 10% points accuracy improvement over baselines on challenging symbolic reasoning benchmarks GSM-symbolic and FOLIO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14567v4">ELOQ: Resources for Enhancing LLM Detection of Out-of-Scope Questions</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Accepted by SIGIR'25
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has become integral to large language models (LLMs), particularly for conversational AI systems where user questions may reference knowledge beyond the LLMs' training cutoff. However, many natural user questions lack well-defined answers, either due to limited domain knowledge or because the retrieval system returns documents that are relevant in appearance but uninformative in content. In such cases, LLMs often produce hallucinated answers without flagging them. While recent work has largely focused on questions with false premises, we study out-of-scope questions, where the retrieved document appears semantically similar to the question but lacks the necessary information to answer it. In this paper, we propose a guided hallucination-based approach ELOQ to automatically generate a diverse set of out-of-scope questions from post-cutoff documents, followed by human verification to ensure quality. We use this dataset to evaluate several LLMs on their ability to detect out-of-scope questions and generate appropriate responses. Finally, we introduce an improved detection method that enhances the reliability of LLM-based question-answering systems in handling out-of-scope questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01980v1">LLM-based Text Simplification and its Effect on User Comprehension and Cognitive Load</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Information on the web, such as scientific publications and Wikipedia, often surpasses users' reading level. To help address this, we used a self-refinement approach to develop a LLM capability for minimally lossy text simplification. To validate our approach, we conducted a randomized study involving 4563 participants and 31 texts spanning 6 broad subject areas: PubMed (biomedical scientific articles), biology, law, finance, literature/philosophy, and aerospace/computer science. Participants were randomized to viewing original or simplified texts in a subject area, and answered multiple-choice questions (MCQs) that tested their comprehension of the text. The participants were also asked to provide qualitative feedback such as task difficulty. Our results indicate that participants who read the simplified text answered more MCQs correctly than their counterparts who read the original text (3.9% absolute increase, p<0.05). This gain was most striking with PubMed (14.6%), while more moderate gains were observed for finance (5.5%), aerospace/computer science (3.8%) domains, and legal (3.5%). Notably, the results were robust to whether participants could refer back to the text while answering MCQs. The absolute accuracy decreased by up to ~9% for both original and simplified setups where participants could not refer back to the text, but the ~4% overall improvement persisted. Finally, participants' self-reported perceived ease based on a simplified NASA Task Load Index was greater for those who read the simplified text (absolute change on a 5-point scale 0.33, p<0.05). This randomized study, involving an order of magnitude more participants than prior works, demonstrates the potential of LLMs to make complex information easier to understand. Our work aims to enable a broader audience to better learn and make use of expert knowledge available on the web, improving information accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v5">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 40 pages, 38 figures An earlier revision of this paper was submitted to ICML. Since then, it has been updated to include new results on training dynamics (4.7) and base models (4.8)
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding. It asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02216v1">LLM-Guided Probabilistic Program Induction for POMDP Model Estimation</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Partially Observable Markov Decision Processes (POMDPs) model decision making under uncertainty. While there are many approaches to approximately solving POMDPs, we aim to address the problem of learning such models. In particular, we are interested in a subclass of POMDPs wherein the components of the model, including the observation function, reward function, transition function, and initial state distribution function, can be modeled as low-complexity probabilistic graphical models in the form of a short probabilistic program. Our strategy to learn these programs uses an LLM as a prior, generating candidate probabilistic programs that are then tested against the empirical distribution and adjusted through feedback. We experiment on a number of classical toy POMDP problems, simulated MiniGrid domains, and two real mobile-base robotics search domains involving partial observability. Our results show that using an LLM to guide in the construction of a low-complexity POMDP model can be more effective than tabular POMDP learning, behavior cloning, or direct LLM planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01638v2">LASSI: An LLM-based Automated Self-Correcting Pipeline for Translating Parallel Scientific Codes</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 8 pages, 1 figure, 7 tables
    </div>
    <details class="paper-abstract">
      This paper addresses the problem of providing a novel approach to sourcing significant training data for LLMs focused on science and engineering. In particular, a crucial challenge is sourcing parallel scientific codes in the ranges of millions to billions of codes. To tackle this problem, we propose an automated pipeline framework called LASSI, designed to translate between parallel programming languages by bootstrapping existing closed- or open-source LLMs. LASSI incorporates autonomous enhancement through self-correcting loops where errors encountered during the compilation and execution of generated code are fed back to the LLM through guided prompting for debugging and refactoring. We highlight the bi-directional translation of existing GPU benchmarks between OpenMP target offload and CUDA to validate LASSI. The results of evaluating LASSI with different application codes across four LLMs demonstrate the effectiveness of LASSI for generating executable parallel codes, with 80% of OpenMP to CUDA translations and 85% of CUDA to OpenMP translations producing the expected output. We also observe approximately 78% of OpenMP to CUDA translations and 62% of CUDA to OpenMP translations execute within 10% of or at a faster runtime than the original benchmark code in the same language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02184v1">Leveraging LLMs to Automate Energy-Aware Refactoring of Parallel Scientific Codes</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used for generating parallel scientific code, most current efforts emphasize functional correctness, often overlooking performance and energy considerations. In this work, we propose LASSI-EE, an automated LLM-based refactoring framework that generates energy-efficient parallel code on a target parallel system for a given parallel code as input. Through a multi-stage, iterative pipeline process, LASSI-EE achieved an average energy reduction of 47% across 85% of the 20 HeCBench benchmarks tested on NVIDIA A100 GPUs. Our findings demonstrate the broader potential of LLMs, not only for generating correct code but also for enabling energy-aware programming. We also address key insights and limitations within the framework, offering valuable guidance for future improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02172v1">Identifying Legal Holdings with LLMs: A Systematic Study of Performance, Scale, and Memorization</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Presented as a short paper at International Conference on Artificial Intelligence and Law 2025 (Chicago, IL)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance in capabilities, it is essential to assess how they perform on established benchmarks. In this study, we present a suite of experiments to assess the performance of modern LLMs (ranging from 3B to 90B+ parameters) on CaseHOLD, a legal benchmark dataset for identifying case holdings. Our experiments demonstrate ``scaling effects'' - performance on this task improves with model size, with more capable models like GPT4o and AmazonNovaPro achieving macro F1 scores of 0.744 and 0.720 respectively. These scores are competitive with the best published results on this dataset, and do not require any technically sophisticated model training, fine-tuning or few-shot prompting. To ensure that these strong results are not due to memorization of judicial opinions contained in the training data, we develop and utilize a novel citation anonymization test that preserves semantic meaning while ensuring case names and citations are fictitious. Models maintain strong performance under these conditions (macro F1 of 0.728), suggesting the performance is not due to rote memorization. These findings demonstrate both the promise and current limitations of LLMs for legal tasks with important implications for the development and measurement of automated legal analytics and legal benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02142v1">Exploring the Potential of Offline RL for Reasoning in LLMs: A Preliminary Study</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Despite significant advances in long-context reasoning by large language models (LLMs), primarily through Online Reinforcement Learning (RL) methods, these approaches incur substantial computational costs and complexity. In contrast, simpler and more economical Offline RL methods remain underexplored. To address this gap, we investigate the effectiveness of Offline RL methods, specifically Direct Preference Optimization (DPO) and its length-desensitized variant LD-DPO, in enhancing the reasoning capabilities of LLMs. Extensive experiments across multiple reasoning benchmarks demonstrate that these simpler Offline RL methods substantially improve model performance, achieving an average enhancement of 3.3\%, with a particularly notable increase of 10.1\% on the challenging Arena-Hard benchmark. Furthermore, we analyze DPO's sensitivity to output length, emphasizing that increasing reasoning length should align with semantic richness, as indiscriminate lengthening may adversely affect model performance. We provide comprehensive descriptions of our data processing and training methodologies, offering empirical evidence and practical insights for developing more cost-effective Offline RL approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02133v1">Enhancing LLM Code Generation: A Systematic Evaluation of Multi-Agent Collaboration and Runtime Debugging for Improved Accuracy, Reliability, and Latency</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) for automated code generation has emerged as a significant focus within AI research. As these pretrained models continue to evolve, their ability to understand and generate complex code structures has opened new possibilities for automating intricate programming tasks for the sake of accurate code generation. Although contemporary foundational models demonstrate promoting results, researchers continue to explore optimal post-training strategies to enhance code quality. These include supervised fine-tuning, retrieval-augmented generation (RAG), debugging, and many others. In this paper, we combine two widely used approaches namely multi-agent collaboration and runtime execution information-based debugging, for improving code generation functionality, reliability, and practical applicability. We perform an empirical study in order to extend the evaluation of the individual strategies as well as the proposed composition of the activities of both strategies. Our study use 19 LLMs to examines the performance of individual and the proposed strategies, offering comprehensive insights into how different programming activities compositions and training paradigms influence code generation effectiveness. In particular, we implement a chained system that combines both strategies to assess their combined impact on functional accuracy, code reliability, and generation latency using two benchmark datasets commonly used for code generation. Our findings provide valuable insights for organizations seeking robust AI-driven coding solutions by guiding them in selecting models that can better adapt to complex post-training strategies, ultimately fostering the adoption of more effective and reliable code generation technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02130v1">Attention Mechanisms Perspective: Exploring LLM Processing of Graph-Structured Data</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 ICML2025 Accept
    </div>
    <details class="paper-abstract">
      Attention mechanisms are critical to the success of large language models (LLMs), driving significant advancements in multiple fields. However, for graph-structured data, which requires emphasis on topological connections, they fall short compared to message-passing mechanisms on fixed links, such as those employed by Graph Neural Networks (GNNs). This raises a question: ``Does attention fail for graphs in natural language settings?'' Motivated by these observations, we embarked on an empirical study from the perspective of attention mechanisms to explore how LLMs process graph-structured data. The goal is to gain deeper insights into the attention behavior of LLMs over graph structures. We uncovered unique phenomena regarding how LLMs apply attention to graph-structured data and analyzed these findings to improve the modeling of such data by LLMs. The primary findings of our research are: 1) While LLMs can recognize graph data and capture text-node interactions, they struggle to model inter-node relationships within graph structures due to inherent architectural constraints. 2) The attention distribution of LLMs across graph nodes does not align with ideal structural patterns, indicating a failure to adapt to graph topology nuances. 3) Neither fully connected attention nor fixed connectivity is optimal; each has specific limitations in its application scenarios. Instead, intermediate-state attention windows improve LLM training performance and seamlessly transition to fully connected windows during inference. Source code: \href{https://github.com/millioniron/LLM_exploration}{LLM4Exploration}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01900v1">CAMOUFLAGE: Exploiting Misinformation Detection Systems Through LLM-driven Adversarial Claim Transformation</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      Automated evidence-based misinformation detection systems, which evaluate the veracity of short claims against evidence, lack comprehensive analysis of their adversarial vulnerabilities. Existing black-box text-based adversarial attacks are ill-suited for evidence-based misinformation detection systems, as these attacks primarily focus on token-level substitutions involving gradient or logit-based optimization strategies, which are incapable of fooling the multi-component nature of these detection systems. These systems incorporate both retrieval and claim-evidence comparison modules, which requires attacks to break the retrieval of evidence and/or the comparison module so that it draws incorrect inferences. We present CAMOUFLAGE, an iterative, LLM-driven approach that employs a two-agent system, a Prompt Optimization Agent and an Attacker Agent, to create adversarial claim rewritings that manipulate evidence retrieval and mislead claim-evidence comparison, effectively bypassing the system without altering the meaning of the claim. The Attacker Agent produces semantically equivalent rewrites that attempt to mislead detectors, while the Prompt Optimization Agent analyzes failed attack attempts and refines the prompt of the Attacker to guide subsequent rewrites. This enables larger structural and stylistic transformations of the text rather than token-level substitutions, adapting the magnitude of changes based on previous outcomes. Unlike existing approaches, CAMOUFLAGE optimizes its attack solely based on binary model decisions to guide its rewriting process, eliminating the need for classifier logits or extensive querying. We evaluate CAMOUFLAGE on four systems, including two recent academic systems and two real-world APIs, with an average attack success rate of 46.92\% while preserving textual coherence and semantic equivalence to the original claims.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20117v2">ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      In this paper we introduce ResearchCodeAgent, a novel multi-agent system leveraging large language models (LLMs) agents to automate the codification of research methodologies described in machine learning literature. The system bridges the gap between high-level research concepts and their practical implementation, allowing researchers auto-generating code of existing research papers for benchmarking or building on top-of existing methods specified in the literature with availability of partial or complete starter code. ResearchCodeAgent employs a flexible agent architecture with a comprehensive action suite, enabling context-aware interactions with the research environment. The system incorporates a dynamic planning mechanism, utilizing both short and long-term memory to adapt its approach iteratively. We evaluate ResearchCodeAgent on three distinct machine learning tasks with distinct task complexity and representing different parts of the ML pipeline: data augmentation, optimization, and data batching. Our results demonstrate the system's effectiveness and generalizability, with 46.9% of generated code being high-quality and error-free, and 25% showing performance improvements over baseline implementations. Empirical analysis shows an average reduction of 57.9% in coding time compared to manual implementation. We observe higher gains for more complex tasks. ResearchCodeAgent represents a significant step towards automating the research implementation process, potentially accelerating the pace of machine learning research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01841v1">Harnessing the Power of LLMs, Informers and Decision Transformers for Intent-driven RAN Management in 6G</a></div>
    <div class="paper-meta">
      📅 2025-05-03
      | 💬 Currently under review
    </div>
    <details class="paper-abstract">
      Intent-driven network management is critical for managing the complexity of 5G and 6G networks. It enables adaptive, on-demand management of the network based on the objectives of the network operators. In this paper, we propose an innovative three-step framework for intent-driven network management based on Generative AI (GenAI) algorithms. First, we fine-tune a Large Language Model (LLM) on a custom dataset using a Quantized Low-Rank Adapter (QLoRA) to enable memory-efficient intent processing within limited computational resources. A Retrieval Augmented Generation (RAG) module is included to support dynamic decision-making. Second, we utilize a transformer architecture for time series forecasting to predict key parameters, such as power consumption, traffic load, and packet drop rate, to facilitate intent validation proactively. Lastly, we introduce a Hierarchical Decision Transformer with Goal Awareness (HDTGA) to optimize the selection and orchestration of network applications and hence, optimize the network. Our intent guidance and processing approach improves BERTScore by 6% and the semantic similarity score by 9% compared to the base LLM model. Again, the proposed predictive intent validation approach can successfully rule out the performance-degrading intents with an average of 88% accuracy. Finally, compared to the baselines, the proposed HDTGA algorithm increases throughput at least by 19.3%, reduces delay by 48.5%, and boosts energy efficiency by 54.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01834v1">Model Context Protocol-based Internet of Experts For Wireless Environment-aware LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong general-purpose reasoning abilities but lack access to wireless environment information due to the absence of native sensory input and domain-specific priors. Previous attempts to apply LLMs in wireless systems either depend on retraining with network-specific data, which compromises language generalization, or rely on manually scripted interfaces, which hinder scalability. To overcome these limitations, we propose a Model Context Protocol (MCP)-based Internet of Experts (IoX) framework that equips LLMs with wireless environment-aware reasoning capabilities. The framework incorporates a set of lightweight expert models, each trained to solve a specific deterministic task in wireless communications, such as detecting a specific wireless attribute, e.g., line-of-sight propagation, Doppler effects, or fading conditions. Through MCP, the LLM can selectively query and interpret expert outputs at inference time, without modifying its own parameters. This architecture enables modular, extensible, and interpretable reasoning over wireless contexts. Evaluated across multiple mainstream LLMs, the proposed wireless environment-aware LLM agents achieve 40%-50% improvements in classification tasks over LLM-only baselines. More broadly, the MCP-based design offers a viable paradigm for future LLMs to inherit structured wireless network management capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01743v1">An LLM-Empowered Low-Resolution Vision System for On-Device Human Behavior Understanding</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      The rapid advancements in Large Vision Language Models (LVLMs) offer the potential to surpass conventional labeling by generating richer, more detailed descriptions of on-device human behavior understanding (HBU) in low-resolution vision systems, such as depth, thermal, and infrared. However, existing large vision language model (LVLM) approaches are unable to understand low-resolution data well as they are primarily designed for high-resolution data, such as RGB images. A quick fixing approach is to caption a large amount of low-resolution data, but it requires a significant amount of labor-intensive annotation efforts. In this paper, we propose a novel, labor-saving system, Llambda, designed to support low-resolution HBU. The core idea is to leverage limited labeled data and a large amount of unlabeled data to guide LLMs in generating informative captions, which can be combined with raw data to effectively fine-tune LVLM models for understanding low-resolution videos in HBU. First, we propose a Contrastive-Oriented Data Labeler, which can capture behavior-relevant information from long, low-resolution videos and generate high-quality pseudo labels for unlabeled data via contrastive learning. Second, we propose a Physical-Knowledge Guided Captioner, which utilizes spatial and temporal consistency checks to mitigate errors in pseudo labels. Therefore, it can improve LLMs' understanding of sequential data and then generate high-quality video captions. Finally, to ensure on-device deployability, we employ LoRA-based efficient fine-tuning to adapt LVLMs for low-resolution data. We evaluate Llambda using a region-scale real-world testbed and three distinct low-resolution datasets, and the experiments show that Llambda outperforms several state-of-the-art LVLM systems up to $40.03\%$ on average Bert-Score.
    </details>
</div>
