# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08631v1">Ensemble based approach to quantifying uncertainty of LLM based classifications</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The output of Large Language Models (LLMs) are a function of the internal model's parameters and the input provided into the context window. The hypothesis presented here is that under a greedy sampling strategy the variance in the LLM's output is a function of the conceptual certainty embedded in the model's parametric knowledge, as well as the lexical variance in the input. Finetuning the model results in reducing the sensitivity of the model output to the lexical input variations. This is then applied to a classification problem and a probabilistic method is proposed for estimating the certainties of the predicted classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08599v1">SPeCtrum: A Grounded Framework for Multidimensional Identity Representation in LLM-Based Agent</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 21 pages, 8 figures, 5 tables, Accepted in NAACL2025 Main
    </div>
    <details class="paper-abstract">
      Existing methods for simulating individual identities often oversimplify human complexity, which may lead to incomplete or flattened representations. To address this, we introduce SPeCtrum, a grounded framework for constructing authentic LLM agent personas by incorporating an individual's multidimensional self-concept. SPeCtrum integrates three core components: Social Identity (S), Personal Identity (P), and Personal Life Context (C), each contributing distinct yet interconnected aspects of identity. To evaluate SPeCtrum's effectiveness in identity representation, we conducted automated and human evaluations. Automated evaluations using popular drama characters showed that Personal Life Context (C)-derived from short essays on preferences and daily routines-modeled characters' identities more effectively than Social Identity (S) and Personal Identity (P) alone and performed comparably to the full SPC combination. In contrast, human evaluations involving real-world individuals found that the full SPC combination provided a more comprehensive self-concept representation than C alone. Our findings suggest that while C alone may suffice for basic identity simulation, integrating S, P, and C enhances the authenticity and accuracy of real-world identity representation. Overall, SPeCtrum offers a structured approach for simulating individuals in LLM agents, enabling more personalized human-AI interactions and improving the realism of simulation-based behavioral studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08586v1">Commercial LLM Agents Are Already Vulnerable to Simple Yet Dangerous Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      A high volume of recent ML security literature focuses on attacks against aligned large language models (LLMs). These attacks may extract private information or coerce the model into producing harmful outputs. In real-world deployments, LLMs are often part of a larger agentic pipeline including memory systems, retrieval, web access, and API calling. Such additional components introduce vulnerabilities that make these LLM-powered agents much easier to attack than isolated LLMs, yet relatively little work focuses on the security of LLM agents. In this paper, we analyze security and privacy vulnerabilities that are unique to LLM agents. We first provide a taxonomy of attacks categorized by threat actors, objectives, entry points, attacker observability, attack strategies, and inherent vulnerabilities of agent pipelines. We then conduct a series of illustrative attacks on popular open-source and commercial agents, demonstrating the immediate practical implications of their vulnerabilities. Notably, our attacks are trivial to implement and require no understanding of machine learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05905v5">Truthful Aggregation of LLMs with an Application to Online Advertising</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The next frontier of online advertising is revenue generation from LLM-generated content. We consider a setting where advertisers aim to influence the responses of an LLM to align with their interests, while platforms seek to maximize advertiser value and ensure user satisfaction. The challenge is that advertisers' preferences generally conflict with those of the user, and advertisers may misreport their preferences. To address this, we introduce MOSAIC, an auction mechanism that ensures that truthful reporting is a dominant strategy for advertisers and that aligns the utility of each advertiser with their contribution to social welfare. Importantly, the mechanism operates without LLM fine-tuning or access to model weights and provably converges to the output of the optimally fine-tuned LLM as computational resources increase. Additionally, it can incorporate contextual information about advertisers, which significantly improves social welfare. Through experiments with a publicly available LLM, we show that MOSAIC leads to high advertiser value and platform revenue with low computational overhead. While our motivating application is online advertising, our mechanism can be applied in any setting with monetary transfers, making it a general-purpose solution for truthfully aggregating the preferences of self-interested agents over LLM-generated replies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08550v1">LLMs can implicitly learn from mistakes in-context</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Learning from mistakes is a fundamental feature of human intelligence. Previous work has shown that Large Language Models (LLMs) can also learn from incorrect answers when provided with a comprehensive rationale detailing why an answer is wrong or how to correct it. In this work, we examine whether LLMs can learn from mistakes in mathematical reasoning tasks when these explanations are not provided. We investigate if LLMs are able to implicitly infer such rationales simply from observing both incorrect and correct answers. Surprisingly, we find that LLMs perform better, on average, when rationales are eliminated from the context and incorrect answers are simply shown alongside correct ones. This approach also substantially outperforms chain-of-thought prompting in our evaluations. We show that these results are consistent across LLMs of different sizes and varying reasoning abilities. Further, we carry out an in-depth analysis, and show that prompting with both wrong and correct answers leads to greater performance and better generalisation than introducing additional, more diverse question-answer pairs into the context. Finally, we show that new rationales generated by models that have only observed incorrect and correct answers are scored equally as highly by humans as those produced with the aid of exemplar rationales. Our results demonstrate that LLMs are indeed capable of in-context implicit learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08515v1">The Paradox of Stochasticity: Limited Creativity and Computational Decoupling in Temperature-Varied LLM Outputs of Structured Fictional Data</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      This study examines how temperature settings and model architectures affect the generation of structured fictional data (names, birthdates) across three large language models (LLMs): llama3.1:8b, deepseek-r1:8b, and mistral:latest. By systematically testing temperature values from 0.0 to 1.0 in increments of 0.1, we conducted 330 trials yielding 889 structured entities, validated for syntactic consistency. Key findings reveal that model architecture significantly influences computational efficiency, with mistral:latest and llama3.1:8b processing data 8x faster than deepseek-r1:8b. Contrary to expectations, temperature showed no correlation with processing time, challenging assumptions about stochastic sampling costs. Output diversity remained limited, as models consistently defaulted to common name archetypes (e.g., 'John Doe' and 'Jane Smith') across all temperatures, though rare names clustered at intermediate values (0.3-0.7). These results demonstrate that architectural optimizations, rather than temperature adjustments, dominate performance in structured generation tasks. The findings emphasize prioritizing model selection over hyperparameter tuning for efficiency and suggest explicit diversity constraints are necessary to mitigate default output biases in synthetic data pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08503v1">Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      In this work, we identify the "2D-Cheating" problem in 3D LLM evaluation, where these tasks might be easily solved by VLMs with rendered images of point clouds, exposing ineffective evaluation of 3D LLMs' unique 3D capabilities. We test VLM performance across multiple 3D LLM benchmarks and, using this as a reference, propose principles for better assessing genuine 3D understanding. We also advocate explicitly separating 3D abilities from 1D or 2D aspects when evaluating 3D LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08395v1">IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are helping millions of users write texts about diverse issues, and in doing so expose users to different ideas and perspectives. This creates concerns about issue bias, where an LLM tends to present just one perspective on a given issue, which in turn may influence how users think about this issue. So far, it has not been possible to measure which issue biases LLMs actually manifest in real user interactions, making it difficult to address the risks from biased LLMs. Therefore, we create IssueBench: a set of 2.49m realistic prompts for measuring issue bias in LLM writing assistance, which we construct based on 3.9k templates (e.g. "write a blog about") and 212 political issues (e.g. "AI regulation") from real user interactions. Using IssueBench, we show that issue biases are common and persistent in state-of-the-art LLMs. We also show that biases are remarkably similar across models, and that all models align more with US Democrat than Republican voter opinion on a subset of issues. IssueBench can easily be adapted to include other issues, templates, or tasks. By enabling robust and realistic measurement, we hope that IssueBench can bring a new quality of evidence to ongoing discussions about LLM biases and how to address them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08381v1">The MoE-Empowered Edge LLMs Deployment: Architecture, Challenges, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 7pages, 1 table, 6 figures
    </div>
    <details class="paper-abstract">
      The powerfulness of LLMs indicates that deploying various LLMs with different scales and architectures on end, edge, and cloud to satisfy different requirements and adaptive heterogeneous hardware is the critical way to achieve ubiquitous intelligence for 6G. However, the massive parameter scale of LLMs poses significant challenges in deploying them on edge devices due to high computational and storage demands. Considering that the sparse activation in Mixture of Experts (MoE) is effective on scalable and dynamic allocation of computational and communications resources at the edge, this paper proposes a novel MoE-empowered collaborative deployment framework for edge LLMs, denoted as CoEL. This framework fully leverages the properties of MoE architecture and encompasses four key aspects: Perception, Deployment, Compression, and Updating. Edge servers broadcast their resource status and the specific resource requirements of LLMs to their neighbors. Then, utilizing this data, two sophisticated deployment strategies are proposed for satisfying varying model scales, ensuring that each model is deployed effectively. One for deploying LLMs on a single edge device through intra-device resource collaboration, and another for a distributed deployment across multiple edge devices via inter-device resource collaboration. Furthermore, both the models and the intermediate data are compressed for reducing memory footprint by quantization and reducing the volume of intermediate data by token fusion and pruning. Finally, given the dynamic of network topology, resource status, and user requirements, the deployment strategies are regularly updated to maintain its relevance and effectiveness. This paper also delineates the challenges and potential research directions for the deployment of edge LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18652v7">$C^2$: Scalable Auto-Feedback for LLM-based Chart Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 NAACL 2025 Main (Long)
    </div>
    <details class="paper-abstract">
      Generating high-quality charts with Large Language Models (LLMs) presents significant challenges due to limited data and the high cost of scaling through human curation. $\langle \text{instruction}, \text{data}, \text{code} \rangle$ triplets are scarce and expensive to manually curate as their creation demands technical expertise. To address this scalability challenge, we introduce a reference-free automatic feedback generator, which eliminates the need for costly human intervention. Our novel framework, C$^2$, consists of (1) an automatic feedback provider (ChartAF) and (2) a diverse, reference-free dataset (ChartUIE-8K). The results are compelling: in our first experiment, 74% of respondents strongly preferred, and 10% preferred, the results after feedback. The second post-feedback experiment demonstrates that ChartAF outperform nine baselines. Moreover, ChartUIE-8K significantly improves data diversity by increasing queries, datasets, and chart types by 5982%, 1936%, and 91%, respectively, over benchmarks. Finally, a study of LLM users revealed that 94% of participants preferred ChartUIE-8K's queries, with 93% deeming them aligned with real-world use cases. Core contributions are available as open-source at chartsquared.github.io, with ample qualitative examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08353v1">Trustworthy GNNs with LLMs: A Systematic Review and Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Submitted to IJCAI 2025
    </div>
    <details class="paper-abstract">
      With the extensive application of Graph Neural Networks (GNNs) across various domains, their trustworthiness has emerged as a focal point of research. Some existing studies have shown that the integration of large language models (LLMs) can improve the semantic understanding and generation capabilities of GNNs, which in turn improves the trustworthiness of GNNs from various aspects. Our review introduces a taxonomy that offers researchers a clear framework for comprehending the principles and applications of different methods and helps clarify the connections and differences among various approaches. Then we systematically survey representative approaches along the four categories of our taxonomy. Through our taxonomy, researchers can understand the applicable scenarios, potential advantages, and limitations of each approach for the the trusted integration of GNNs with LLMs. Finally, we present some promising directions of work and future trends for the integration of LLMs and GNNs to improve model trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08332v1">Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs) has raised concerns about potential misuse. One practical solution is to embed a watermark in the text, allowing ownership verification through watermark extraction. Existing methods primarily focus on defending against modification attacks, often neglecting other spoofing attacks. For example, attackers can alter the watermarked text to produce harmful content without compromising the presence of the watermark, which could lead to false attribution of this malicious content to the LLM. This situation poses a serious threat to the LLMs service providers and highlights the significance of achieving modification detection and generated-text detection simultaneously. Therefore, we propose a technique to detect modifications in text for unbiased watermark which is sensitive to modification. We introduce a new metric called ``discarded tokens", which measures the number of tokens not included in watermark detection. When a modification occurs, this metric changes and can serve as evidence of the modification. Additionally, we improve the watermark detection process and introduce a novel method for unbiased watermark. Our experiments demonstrate that we can achieve effective dual detection capabilities: modification detection and generated-text detection by watermark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08312v1">Word Synchronization Challenge: A Benchmark for Word Association Responses for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      This paper introduces the Word Synchronization Challenge, a novel benchmark to evaluate large language models (LLMs) in Human-Computer Interaction (HCI). This benchmark uses a dynamic game-like framework to test LLMs ability to mimic human cognitive processes through word associations. By simulating complex human interactions, it assesses how LLMs interpret and align with human thought patterns during conversational exchanges, which are essential for effective social partnerships in HCI. Initial findings highlight the influence of model sophistication on performance, offering insights into the models capabilities to engage in meaningful social interactions and adapt behaviors in human-like ways. This research advances the understanding of LLMs potential to replicate or diverge from human cognitive functions, paving the way for more nuanced and empathetic human-machine collaborations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08298v1">Improving Existing Optimization Algorithms with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into optimization has created a powerful synergy, opening exciting research opportunities. This paper investigates how LLMs can enhance existing optimization algorithms. Using their pre-trained knowledge, we demonstrate their ability to propose innovative heuristic variations and implementation strategies. To evaluate this, we applied a non-trivial optimization algorithm, Construct, Merge, Solve and Adapt (CMSA) -- a hybrid metaheuristic for combinatorial optimization problems that incorporates a heuristic in the solution construction phase. Our results show that an alternative heuristic proposed by GPT-4o outperforms the expert-designed heuristic of CMSA, with the performance gap widening on larger and denser graphs. Project URL: https://imp-opt-algo-llms.surge.sh/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03824v2">Syntriever: How to Train Your Retriever with Synthetic Data from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings, Accepted
    </div>
    <details class="paper-abstract">
      LLMs have boosted progress in many AI applications. Recently, there were attempts to distill the vast knowledge of LLMs into information retrieval systems. Those distillation methods mostly use output probabilities of LLMs which are unavailable in the latest black-box LLMs. We propose Syntriever, a training framework for retrievers using synthetic data from black-box LLMs. Syntriever consists of two stages. Firstly in the distillation stage, we synthesize relevant and plausibly irrelevant passages and augmented queries using chain-of-thoughts for the given queries. LLM is asked to self-verify the synthetic data for possible hallucinations, after which retrievers are trained with a loss designed to cluster the embeddings of relevant passages. Secondly in the alignment stage, we align the retriever with the preferences of LLMs. We propose a preference modeling called partial Plackett-Luce ranking to learn LLM preferences with regularization which prevents the model from deviating excessively from that trained in the distillation stage. Experiments show that Syntriever achieves state-of-the-art performances on benchmark datasets from various domains in nDCG@$K$. The code is available at \href{https://github.com/kmswin1/Syntriever}{https://github.com/kmswin1/Syntriever}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08271v1">MoLoRec: A Generalizable and Efficient Framework for LLM-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in recent years, owing to their impressive generalization capabilities and rich world knowledge. To capitalize on the potential of using LLMs as recommender systems, mainstream approaches typically focus on two paradigms. The first paradigm designs multi-domain or multi-task instruction data for generalizable recommendation, so as to align LLMs with general recommendation areas and deal with cold-start recommendation. The second paradigm enhances domain-specific recommendation tasks with parameter-efficient fine-tuning techniques, in order to improve models under the warm recommendation scenarios. While most previous works treat these two paradigms separately, we argue that they have complementary advantages, and combining them together would be helpful. To that end, in this paper, we propose a generalizable and efficient LLM-based recommendation framework MoLoRec. Our approach starts by parameter-efficient fine-tuning a domain-general module with general recommendation instruction data, to align LLM with recommendation knowledge. Then, given users' behavior of a specific domain, we construct a domain-specific instruction dataset and apply efficient fine-tuning to the pre-trained LLM. After that, we provide approaches to integrate the above domain-general part and domain-specific part with parameters mixture. Please note that, MoLoRec is efficient with plug and play, as the domain-general module is trained only once, and any domain-specific plug-in can be efficiently merged with only domain-specific fine-tuning. Extensive experiments on multiple datasets under both warm and cold-start recommendation scenarios validate the effectiveness and generality of the proposed MoLoRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08224v1">Flow-of-Action: SOP Enhanced LLM-Based Multi-Agent System for Root Cause Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Accepted by WWW'25 Industry Track
    </div>
    <details class="paper-abstract">
      In the realm of microservices architecture, the occurrence of frequent incidents necessitates the employment of Root Cause Analysis (RCA) for swift issue resolution. It is common that a serious incident can take several domain experts hours to identify the root cause. Consequently, a contemporary trend involves harnessing Large Language Models (LLMs) as automated agents for RCA. Though the recent ReAct framework aligns well with the Site Reliability Engineers (SREs) for its thought-action-observation paradigm, its hallucinations often lead to irrelevant actions and directly affect subsequent results. Additionally, the complex and variable clues of the incident can overwhelm the model one step further. To confront these challenges, we propose Flow-of-Action, a pioneering Standard Operation Procedure (SOP) enhanced LLM-based multi-agent system. By explicitly summarizing the diagnosis steps of SREs, SOP imposes constraints on LLMs at crucial junctures, guiding the RCA process towards the correct trajectory. To facilitate the rational and effective utilization of SOPs, we design an SOP-centric framework called SOP flow. SOP flow contains a series of tools, including one for finding relevant SOPs for incidents, another for automatically generating SOPs for incidents without relevant ones, and a tool for converting SOPs into code. This significantly alleviates the hallucination issues of ReAct in RCA tasks. We also design multiple auxiliary agents to assist the main agent by removing useless noise, narrowing the search space, and informing the main agent whether the RCA procedure can stop. Compared to the ReAct method's 35.50% accuracy, our Flow-of-Action method achieves 64.01%, meeting the accuracy requirements for RCA in real-world systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07709v2">MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Open-ended learning agents must efficiently prioritize goals in vast possibility spaces, focusing on those that maximize learning progress (LP). When such autotelic exploration is achieved by LLM agents trained with online RL in high-dimensional and evolving goal spaces, a key challenge for LP prediction is modeling one's own competence, a form of metacognitive monitoring. Traditional approaches either require extensive sampling or rely on brittle expert-defined goal groupings. We introduce MAGELLAN, a metacognitive framework that lets LLM agents learn to predict their competence and LP online. By capturing semantic relationships between goals, MAGELLAN enables sample-efficient LP estimation and dynamic adaptation to evolving goal spaces through generalization. In an interactive learning environment, we show that MAGELLAN improves LP prediction efficiency and goal prioritization, being the only method allowing the agent to fully master a large and evolving goal space. These results demonstrate how augmenting LLM agents with a metacognitive ability for LP predictions can effectively scale curriculum learning to open-ended goal spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08213v1">LLM Modules: Knowledge Transfer from a Large to a Small Model using Enhanced Cross-Attention</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Code and pre-trained weights available at https://huggingface.co/kkolomeitsev/llm-modules
    </div>
    <details class="paper-abstract">
      In this work, we propose an architecture of LLM Modules that enables the transfer of knowledge from a large pre-trained model to a smaller model using an Enhanced Cross-Attention mechanism. In the proposed scheme, the Qwen2-1.5B model is frozen and its representations are passed through specially designed attention layers to the GPT-Neo-125M model, which is trained on limited computational resources. Experimental results on the Bespoke-Stratos-17k dataset demonstrate that after 15 epochs of training, the combined model generates responses comparable in quality to those obtained by distillation. We discuss the advantages of the modular approach, provide examples of input queries and comparative analysis, and outline prospects for further extension of the method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08180v1">Enhancing LLM Character-Level Manipulation via Divide and Conquer</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong generalization capabilities across a wide range of natural language processing (NLP) tasks. However, they exhibit notable weaknesses in character-level string manipulation, struggling with fundamental operations such as character deletion, insertion, and substitution. These challenges stem primarily from tokenization constraints, despite the critical role of such operations in data preprocessing and code generation. Through systematic analysis, we derive two key insights: (1) LLMs face significant difficulties in leveraging intrinsic token knowledge for character-level reasoning, and (2) atomized word structures can substantially enhance LLMs' ability to process token-level structural information. Building on these insights, we propose Character-Level Manipulation via Divide and Conquer, a novel approach designed to bridge the gap between token-level processing and character-level manipulation. Our method decomposes complex operations into explicit character-level subtasks coupled with controlled token reconstruction phases, leading to significant improvements in accuracy. Without additional training, our method significantly improves accuracies on the $\texttt{Deletion}$, $\texttt{Insertion}$, and $\texttt{Substitution}$ tasks. To support further research, we open-source our implementation and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08177v1">SycEval: Evaluating LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in educational, clinical, and professional settings, but their tendency for sycophancy -- prioritizing user agreement over independent reasoning -- poses risks to reliability. This study introduces a framework to evaluate sycophantic behavior in ChatGPT-4o, Claude-Sonnet, and Gemini-1.5-Pro across AMPS (mathematics) and MedQuad (medical advice) datasets. Sycophantic behavior was observed in 58.19% of cases, with Gemini exhibiting the highest rate (62.47%) and ChatGPT the lowest (56.71%). Progressive sycophancy, leading to correct answers, occurred in 43.52% of cases, while regressive sycophancy, leading to incorrect answers, was observed in 14.66%. Preemptive rebuttals demonstrated significantly higher sycophancy rates than in-context rebuttals (61.75% vs. 56.52%, $Z=5.87$, $p<0.001$), particularly in computational tasks, where regressive sycophancy increased significantly (preemptive: 8.13%, in-context: 3.54%, $p<0.001$). Simple rebuttals maximized progressive sycophancy ($Z=6.59$, $p<0.001$), while citation-based rebuttals exhibited the highest regressive rates ($Z=6.59$, $p<0.001$). Sycophantic behavior showed high persistence (78.5%, 95% CI: [77.2%, 79.8%]) regardless of context or model. These findings emphasize the risks and opportunities of deploying LLMs in structured and dynamic domains, offering insights into prompt programming and model optimization for safer AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20002v3">The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 This work was submitted for review on Sept. 5, 2024, and the initial version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects the up-to-date experimental results
    </div>
    <details class="paper-abstract">
      The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08145v1">Democratizing AI: Open-source Scalable LLM Training on GPU-based Supercomputers</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Training and fine-tuning large language models (LLMs) with hundreds of billions to trillions of parameters requires tens of thousands of GPUs, and a highly scalable software stack. In this work, we present a novel four-dimensional hybrid parallel algorithm implemented in a highly scalable, portable, open-source framework called AxoNN. We describe several performance optimizations in AxoNN to improve matrix multiply kernel performance, overlap non-blocking collectives with computation, and performance modeling to choose performance optimal configurations. These have resulted in unprecedented scaling and peak flop/s (bf16) for training of GPT-style transformer models on Perlmutter (620.1 Petaflop/s), Frontier (1.381 Exaflop/s) and Alps (1.423 Exaflop/s). While the abilities of LLMs improve with the number of trainable parameters, so do privacy and copyright risks caused by memorization of training data, which can cause disclosure of sensitive or private information at inference time. We highlight this side effect of scale through experiments that explore "catastrophic memorization", where models are sufficiently large to memorize training data in a single pass, and present an approach to prevent it. As part of this study, we demonstrate fine-tuning of a 405-billion parameter LLM using AxoNN on Frontier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08142v1">Bridging the Safety Gap: A Guardrail Pipeline for Trustworthy LLM Inferences</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 arXiv admin note: text overlap with arXiv:2406.10847
    </div>
    <details class="paper-abstract">
      We present Wildflare GuardRail, a guardrail pipeline designed to enhance the safety and reliability of Large Language Model (LLM) inferences by systematically addressing risks across the entire processing workflow. Wildflare GuardRail integrates several core functional modules, including Safety Detector that identifies unsafe inputs and detects hallucinations in model outputs while generating root-cause explanations, Grounding that contextualizes user queries with information retrieved from vector databases, Customizer that adjusts outputs in real time using lightweight, rule-based wrappers, and Repairer that corrects erroneous LLM outputs using hallucination explanations provided by Safety Detector. Results show that our unsafe content detection model in Safety Detector achieves comparable performance with OpenAI API, though trained on a small dataset constructed with several public datasets. Meanwhile, the lightweight wrappers can address malicious URLs in model outputs in 1.06s per query with 100% accuracy without costly model calls. Moreover, the hallucination fixing model demonstrates effectiveness in reducing hallucinations with an accuracy of 80.7%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08141v1">LowRA: Accurate and Efficient LoRA Fine-Tuning of LLMs under 2 Bits</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) is increasingly costly as models scale to hundreds of billions of parameters, and even parameter-efficient fine-tuning (PEFT) methods like LoRA remain resource-intensive. We introduce LowRA, the first framework to enable LoRA fine-tuning below 2 bits per parameter with minimal performance loss. LowRA optimizes fine-grained quantization - mapping, threshold selection, and precision assignment - while leveraging efficient CUDA kernels for scalable deployment. Extensive evaluations across 4 LLMs and 4 datasets show that LowRA achieves a superior performance-precision trade-off above 2 bits and remains accurate down to 1.15 bits, reducing memory usage by up to 50%. Our results highlight the potential of ultra-low-bit LoRA fine-tuning for resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14654v2">MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have demonstrated significant advancements, particularly in their ability to serve as agents thereby surpassing their traditional role as chatbots. These agents can leverage their planning and tool utilization capabilities to address tasks specified at a high level. However, a standardized dataset to benchmark the agent capabilities of LLMs in medical applications is currently lacking, making the evaluation of LLMs on complex tasks in interactive healthcare environments challenging. To address this gap, we introduce MedAgentBench, a broad evaluation suite designed to assess the agent capabilities of large language models within medical records contexts. MedAgentBench encompasses 300 patient-specific clinically-derived tasks from 10 categories written by human physicians, realistic profiles of 100 patients with over 700,000 data elements, a FHIR-compliant interactive environment, and an accompanying codebase. The environment uses the standard APIs and communication infrastructure used in modern EMR systems, so it can be easily migrated into live EMR systems. MedAgentBench presents an unsaturated agent-oriented benchmark that current state-of-the-art LLMs exhibit some ability to succeed at. The best model (Claude 3.5 Sonnet v2) achieves a success rate of 69.67%. However, there is still substantial space for improvement which gives the community a next direction to optimize. Furthermore, there is significant variation in performance across task categories. MedAgentBench establishes this and is publicly available at https://github.com/stanfordmlgroup/MedAgentBench , offering a valuable framework for model developers to track progress and drive continuous improvements in the agent capabilities of large language models within the medical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08127v1">Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Ongoing work, 13 pages, 2 figures, 3 Tables
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown strong general reasoning abilities, yet their effectiveness in financial reasoning remains underexplored. In this study, we comprehensively evaluate 16 powerful reasoning and general LLMs on three complex financial tasks involving financial text, tabular data, and equations, assessing numerical reasoning, tabular interpretation, financial terminology comprehension, long-context processing, and equation-based problem solving. Our results show that while better datasets and pretraining improve financial reasoning, general enhancements like CoT fine-tuning do not always yield consistent gains. Moreover, all reasoning strategies face challenges in improving performance on long-context and multi-table tasks. To address these limitations, we develop a financial reasoning-enhanced model based on Llama-3.1-8B-Instruct, by CoT fine-tuning and reinforcement learning with domain-specific reasoning paths. Even with simple fine-tuning with one financial dataset, our model achieves a consistent 10% performance improvement across tasks, surpassing all 8B models and even Llama3-70B-Instruct and Llama3.1-70B-Instruct on average. Our results highlight the need for domain-specific adaptations in financial tasks, emphasizing future directions such as multi-table reasoning, long-context processing, and financial terminology comprehension. All our datasets, models, and codes are publicly available. Furthermore, we introduce a leaderboard for benchmarking future datasets and models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04961v2">Demystifying Domain-adaptive Post-training for Financial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Domain-adaptive post-training of large language models (LLMs) has emerged as a promising approach for specialized domains such as medicine and finance. However, significant challenges remain in identifying optimal adaptation criteria and training strategies across varying data and model configurations. To address these challenges, we introduce FINDAP, a systematic and fine-grained investigation into domain adaptive post-training of LLMs for the finance domain. Our approach consists of four key components: FinCap, which defines the core capabilities required for the target domain; FinRec, an effective training recipe that jointly optimizes continual pre-training and instruction-following, along with a novel preference data distillation method leveraging process signals from a generative reward model; FinTrain, a curated set of training datasets supporting FinRec; and FinEval, a comprehensive evaluation suite aligned with FinCap. The resulting model, Llama-Fin, achieves state-of-the-art performance across a wide range of financial tasks. Our analysis also highlights how each post-training stage contributes to distinct capabilities, uncovering specific challenges and effective solutions, providing valuable insights for domain adaptation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17630v2">Uncertainty Quantification and Decomposition for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 WWW 2025
    </div>
    <details class="paper-abstract">
      Despite the widespread adoption of large language models (LLMs) for recommendation, we demonstrate that LLMs often exhibit uncertainty in their recommendations. To ensure the trustworthy use of LLMs in generating recommendations, we emphasize the importance of assessing the reliability of recommendations generated by LLMs. We start by introducing a novel framework for estimating the predictive uncertainty to quantitatively measure the reliability of LLM-based recommendations. We further propose to decompose the predictive uncertainty into recommendation uncertainty and prompt uncertainty, enabling in-depth analyses of the primary source of uncertainty. Through extensive experiments, we (1) demonstrate predictive uncertainty effectively indicates the reliability of LLM-based recommendations, (2) investigate the origins of uncertainty with decomposed uncertainty measures, and (3) propose uncertainty-aware prompting for a lower predictive uncertainty and enhanced recommendation. Our source code and model weights are available at https://github.com/WonbinKweon/UNC_LLM_REC_WWW2025
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08109v1">HuDEx: Integrating Hallucination Detection and Explainability for Enhancing the Reliability of LLM responses</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have shown promising improvements, often surpassing existing methods across a wide range of downstream tasks in natural language processing. However, these models still face challenges, which may hinder their practical applicability. For example, the phenomenon of hallucination is known to compromise the reliability of LLMs, especially in fields that demand high factual precision. Current benchmarks primarily focus on hallucination detection and factuality evaluation but do not extend beyond identification. This paper proposes an explanation enhanced hallucination-detection model, coined as HuDEx, aimed at enhancing the reliability of LLM-generated responses by both detecting hallucinations and providing detailed explanations. The proposed model provides a novel approach to integrate detection with explanations, and enable both users and the LLM itself to understand and reduce errors. Our measurement results demonstrate that the proposed model surpasses larger LLMs, such as Llama3 70B and GPT-4, in hallucination detection accuracy, while maintaining reliable explanations. Furthermore, the proposed model performs well in both zero-shot and other test environments, showcasing its adaptability across diverse benchmark datasets. The proposed approach further enhances the hallucination detection research by introducing a novel approach to integrating interpretability with hallucination detection, which further enhances the performance and reliability of evaluating hallucinations in language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05006v3">COAST: Enhancing the Code Debugging Ability of LLMs through Communicative Agent Based Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Code debugging is a vital stage of software development, essential for ensuring the reliability and performance of Large Language Models (LLMs) in the code generation task. Human debugging typically follows a multi-stage process, which includes Bug Localization, Bug Identification, Code Repair, and Code Recognition. However, existing code debugging benchmarks predominantly focus on the Code Repair stage, which offers only a limited perspective on evaluating the debugging capabilities of LLMs. In this paper, we introduce DEBUGEVAL, a comprehensive benchmark for evaluating the debugging abilities of LLMs by emulating the multi-stage human debugging process. Through evaluating on DEBUGEVAL, we observe that 7B-scale models consistently underperform compared to their larger counterparts, highlighting their limitations in comprehending code semantics. In this case, we propose the COmmunicative Agent-based data SynThesis (COAST) framework, which employs a multi-agent system to generate high-quality training data for supervised fine-tuning (SFT). Experimental results demonstrate that COAST-generated data outperform human-curated and GPT-4-generated data, enabling 7B-scale LLMs to achieve debugging performance comparable to GPT-3.5. All data and codes are available at https://github.com/NEUIR/COAST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07191v2">Bag of Tricks for Inference-time Computation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      With the advancement of large language models (LLMs), solving complex reasoning tasks has gained increasing attention. Inference-time computation methods (e.g., Best-of-N, beam search, et al.) are particularly valuable as they can enhance reasoning performance without modifying model parameters or requiring additional training. However, these techniques come with implementation challenges, and most existing methods remain at the proof-of-concept stage with limited practical adoption due to their computational complexity and varying effectiveness across different tasks. In this paper, we investigate and benchmark diverse inference-time computation strategies across reasoning tasks of varying complexity. Since most current methods rely on a proposer-verifier pipeline that first generates candidate solutions (e.g., reasoning solutions) and then selects the best one based on reward signals (e.g., RLHF rewards, process rewards), our research focuses on optimizing both candidate solution generation (e.g., instructing prompts, hyperparameters such as temperature and top-p) and reward mechanisms (e.g., self-evaluation, reward types). Through extensive experiments (more than 20,000 A100-80G GPU hours with over 1,000 experiments) across a variety of models (e.g., Llama, Qwen, and Mistral families) of various sizes, our ablation studies reveal that previously overlooked strategies can significantly enhance performance (e.g., tuning temperature can improve reasoning task performance by up to 5%). Furthermore, we establish a standardized benchmark for inference-time computation by systematically evaluating six representative methods across eight reasoning tasks. These findings provide a stronger foundation for future research. The code is available at https://github.com/usail-hkust/benchmark_inference_time_computation_LL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15512v2">Reverse Question Answering: Can an LLM Write a Question so Hard (or Bad) that it Can't Answer?</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Question answering (QA), giving correct answers to questions, is a popular task, but we test reverse question answering (RQA): for an input answer, give a question with that answer. Past work tests QA and RQA separately, but we test them jointly, comparing their difficulty, aiding benchmark design, and checking reasoning consistency. We run 16 LLMs on QA and RQA with trivia questions/answers, revealing: 1) Versus QA, LLMs are much less accurate in RQA for numerical answers, but slightly more accurate in RQA for textual answers; 2) LLMs often answer their own invalid questions from RQA accurately in QA, so RQA errors are not from knowledge gaps alone; 3) RQA errors correlate with question difficulty and inversely correlate with answer frequencies in the Dolma corpus; and 4) LLMs struggle to provide valid multi-hop questions. By finding question and answer types that lead to RQA errors, we suggest improvements for LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.09113v2">Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which has been shown to successfully jailbreak multiple open-source LLMs. Drawing inspiration from the difficulties of discrete token optimization, our method relaxes the discrete jailbreak optimization into a continuous optimization process while gradually increasing the sparsity of the optimizing vectors. This technique effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than state-of-the-art token-level methods. On Harmbench, our approach achieves the highest attack success rate on seven out of eight LLMs compared to the latest jailbreak methods. Trigger Warning: This paper contains model behavior that can be offensive in nature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14427v3">GraphSOS: Graph Sampling and Order Selection to Help LLMs Understand Graphs Better</a></div>
    <div class="paper-meta">
      📅 2025-02-12
    </div>
    <details class="paper-abstract">
      The success of Large Language Models (LLMs) in various domains has led researchers to apply them to graph-related problems by converting graph data into natural language text. However, unlike graph data, natural language inherently has sequential order. We observe a counter-intuitive fact that when the order of nodes or edges in the natural language description of a graph is shuffled, despite describing the same graph, model performance fluctuates between high performance and random guessing. Additionally, due to LLMs' limited input context length, current methods typically randomly sample neighbors of target nodes as representatives of their neighborhood, which may not always be effective for accurate reasoning. To address these gaps, we introduce GraphSOS (Graph Sampling and Order Selection). This novel model framework features an Order Selector Module to ensure proper serialization order of the graph and a Subgraph Sampling Module to sample subgraphs with better structure for better reasoning. Furthermore, we propose Graph CoT obtained through distillation, and enhance LLM's reasoning and zero-shot learning capabilities for graph tasks through instruction tuning. Experiments on multiple datasets for node classification and graph question-answering demonstrate that GraphSOS improves LLMs' performance and generalization ability on graph tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08045v1">Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      A large number of studies rely on closed-style multiple-choice surveys to evaluate cultural alignment in Large Language Models (LLMs). In this work, we challenge this constrained evaluation paradigm and explore more realistic, unconstrained approaches. Using the World Values Survey (WVS) and Hofstede Cultural Dimensions as case studies, we demonstrate that LLMs exhibit stronger cultural alignment in less constrained settings, where responses are not forced. Additionally, we show that even minor changes, such as reordering survey choices, lead to inconsistent outputs, exposing the limitations of closed-style evaluations. Our findings advocate for more robust and flexible evaluation frameworks that focus on specific cultural proxies, encouraging more nuanced and accurate assessments of cultural alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08037v1">Franken-Adapter: Cross-Lingual Adaptation of LLMs by Embedding Surgery</a></div>
    <div class="paper-meta">
      📅 2025-02-12
      | 💬 33 pages
    </div>
    <details class="paper-abstract">
      The capabilities of Large Language Models (LLMs) in low-resource languages lag far behind those in English, making their universal accessibility a significant challenge. To alleviate this, we present $\textit{Franken-Adapter}$, a modular language adaptation approach for decoder-only LLMs with embedding surgery. Our method begins by creating customized vocabularies for target languages and performing language adaptation through embedding tuning on multilingual data. These pre-trained embeddings are subsequently integrated with LLMs that have been instruction-tuned on English alignment data to enable zero-shot cross-lingual transfer. Our experiments on $\texttt{Gemma2}$ models with up to 27B parameters demonstrate improvements of up to 20% across 96 languages, spanning both discriminative and generative tasks, with minimal regressions ($<$1%) in English. Further in-depth analysis reveals the critical role of customizing tokenizers in enhancing language adaptation, while boosting inference efficiency. Additionally, we show the versatility of our method by achieving a 14% improvement over a math-optimized LLM across 20 languages, offering a modular solution to transfer reasoning abilities across languages post hoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06808v2">Effect of Adaptive Communication Support on LLM-powered Human-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 13 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Effective human-robot collaboration requires robot to adopt their roles and levels of support based on human needs, task requirements, and complexity. Traditional human-robot teaming often relies on a pre-determined robot communication scheme, restricting teamwork adaptability in complex tasks. Leveraging strong communication capabilities of Large Language Models (LLMs), we propose a Human-Robot Teaming Framework with Multi-Modal Language feedback (HRT-ML), a framework designed to enhance human-robot interaction by adjusting the frequency and content of language-based feedback. HRT-ML framework includes two core modules: a Coordinator for high-level, low-frequency strategic guidance, and a Manager for subtask-specific, high-frequency instructions, enabling passive and active interactions with human teammates. To assess the impact of language feedback in collaborative scenarios, we conducted experiments in an enhanced Overcooked environment with varying levels of task complexity (easy, medium, hard) and feedback frequency (inactive, passive, active, superactive). Our results show that as task complexity increases relative to human capabilities, human teammates exhibited a stronger preference towards robotic agents that can offer frequent, proactive support. However, when task complexities exceed the LLM's capacity, noisy and inaccurate feedback from superactive robotic agents can instead hinder team performance, as it requires human teammates to increase their effort to interpret and respond to a large number of communications, with limited performance return. Our results offer a general principle for robotic agents to dynamically adjust their levels and frequencies of communications to work seamlessly with humans and achieve improved teaming performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07752v1">Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Designing efficient optimizers for large language models (LLMs) with low-memory requirements and fast convergence is an important and challenging problem. This paper makes a step towards the systematic design of such optimizers through the lens of structured Fisher information matrix (FIM) approximation. We show that many state-of-the-art efficient optimizers can be viewed as solutions to FIM approximation (under the Frobenius norm) with specific structural assumptions. Building on these insights, we propose two design recommendations of practical efficient optimizers for LLMs, involving the careful selection of structural assumptions to balance generality and efficiency, and enhancing memory efficiency of optimizers with general structures through a novel low-rank extension framework. We demonstrate how to use each design approach by deriving new memory-efficient optimizers: Row and Column Scaled SGD (RACS) and Adaptive low-dimensional subspace estimation (Alice). Experiments on LLaMA pre-training (up to 1B parameters) validate the effectiveness, showing faster and better convergence than existing memory-efficient baselines and Adam with little memory overhead. Notably, Alice achieves better than 2x faster convergence over Adam, while RACS delivers strong performance on the 1B model with SGD-like memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07728v1">Verifying LLM-Generated Code in the Context of Software Verification with Ada/SPARK</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable code generation capabilities, but the correctness of the generated code cannot be inherently trusted. This paper explores the feasibility of using formal software verification, specifically the SPARK framework for Ada, to ensure the reliability of LLM-generated code. We present Marmaragan, a tool that leverages an LLM in order to generate SPARK annotations for existing programs, enabling formal verification of the code. The tool is benchmarked on a curated set of SPARK programs, with annotations selectively removed to test specific capabilities. The performance of Marmaragan with GPT-4o on the benchmark is promising, with correct annotations having been generated for 50.7% of the benchmark cases. The results establish a foundation for future work on combining the power of LLMs with the reliability of formal software verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11779v2">Glinthawk: A Two-Tiered Architecture for Offline LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      We introduce Glinthawk, an architecture for offline Large Language Model (LLM) inference. By leveraging a two-tiered structure, Glinthawk optimizes the utilization of the high-end accelerators ("Tier 1") by offloading the attention mechanism to lower-end compute tier ("Tier 2"). This separation allows the memory demand of the attention, known as the key-value cache, to scale independently from the model weights, enabling larger batch sizes and more efficient accelerator usage. Prototyped with NVIDIA T4 GPUs and standard CPU VMs, Glinthawk improves throughput by $5.9\times$ and reduces cost of generation by $2.8\times$, compared to paged attention baselines. For long sequence lengths, it achieves $16.3\times$ throughput improvement at $2.4\times$ less cost. Our evaluation shows that this architecture can tolerate moderate network latency with minimal performance degradation, making it highly effective for latency-tolerant, throughput-focused applications such as batch processing. The prototype is publicly available at https://github.com/microsoft/glinthawk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07709v1">MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Open-ended learning agents must efficiently prioritize goals in vast possibility spaces, focusing on those that maximize learning progress (LP). When such autotelic exploration is achieved by LLM agents trained with online RL in high-dimensional and evolving goal spaces, a key challenge for LP prediction is modeling one's own competence, a form of metacognitive monitoring. Traditional approaches either require extensive sampling or rely on brittle expert-defined goal groupings. We introduce MAGELLAN, a metacognitive framework that lets LLM agents learn to predict their competence and LP online. By capturing semantic relationships between goals, MAGELLAN enables sample-efficient LP estimation and dynamic adaptation to evolving goal spaces through generalization. In an interactive learning environment, we show that MAGELLAN improves LP prediction efficiency and goal prioritization, being the only method allowing the agent to fully master a large and evolving goal space. These results demonstrate how augmenting LLM agents with a metacognitive ability for LP predictions can effectively scale curriculum learning to open-ended goal spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07698v1">A Framework for LLM-powered Design Assistants</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Design assistants are frameworks, tools or applications intended to facilitate both the creative and technical facets of design processes. Large language models (LLMs) are AI systems engineered to analyze and produce text resembling human language, leveraging extensive datasets. This study introduces a framework wherein LLMs are employed as Design Assistants, focusing on three key modalities within the Design Process: Idea Exploration, Dialogue with Designers, and Design Evaluation. Importantly, our framework is not confined to a singular design process but is adaptable across various processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06556v2">ProjectTest: A Project-level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Unit test generation has become a promising and important use case of LLMs. However, existing evaluation benchmarks for assessing LLM unit test generation capabilities focus on function- or class-level code rather than more practical and challenging project-level codebases. To address such limitation, we propose ProjectTest, a project-level benchmark for unit test generation covering Python, Java, and JavaScript. ProjectTest features 20 moderate-sized and high-quality projects per language. We evaluate nine frontier LLMs on ProjectTest and the results show that all frontier LLMs tested exhibit moderate performance on ProjectTest on Python and Java, highlighting the difficulty of ProjectTest. We also conduct a thorough error analysis, which shows that even frontier LLMs, such as Claude-3.5-Sonnet, have significant simple errors, including compilation and cascade errors. Motivated by this observation, we further evaluate all frontier LLMs under manual error-fixing and self-error-fixing scenarios to assess their potential when equipped with error-fixing mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08770v2">Model Surgery: Modulating LLM's Behavior Via Simple Parameter Editing</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 23 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential as generalist assistants, showcasing powerful task understanding and problem-solving capabilities. To deploy LLMs as AI assistants, it is crucial that these models exhibit desirable behavioral traits, such as non-toxicity and resilience against jailbreak attempts. Current approaches for detoxification or preventing jailbreaking usually involve Supervised Fine-Tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF), which requires finetuning billions of parameters through gradient descent with substantial computational cost. Furthermore, models modified through SFT and RLHF may deviate from the pretrained models, potentially leading to a degradation in foundational LLM capabilities. In this paper, we observe that surprisingly, directly editing a small subset of parameters can effectively modulate specific behaviors of LLMs, such as detoxification and resistance to jailbreaking, with only inference-level computational resources. Experiments demonstrate that in the detoxification task, our approach achieves reductions of up to 90.0% in toxicity on the RealToxicityPrompts dataset and 49.2% on ToxiGen, while maintaining the LLM's general capabilities in areas such as common sense, question answering, and mathematics
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04964v2">CoCoA: A Generalized Approach to Uncertainty Quantification by Integrating Confidence and Consistency of LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompasses a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches and shown impressive performance in various applications. However, they sometimes fail to outperform much simpler baseline methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency that leads to a family of efficient and robust UQ methods. We evaluate our approach across a variety of tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00696v3">Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Rating-based human evaluation has become an essential tool to accurately evaluate the impressive performance of large language models (LLMs). However, current rating systems suffer from several important limitations: first, they fail to account for biases that significantly influence evaluation results, second, they require large and expensive preference datasets to obtain accurate ratings, and third, they do not facilitate meaningful comparisons of model ratings across different tasks. To address these issues, we introduce Polyrating, an expressive and flexible rating system based on maximum a posteriori estimation that enables a more nuanced and thorough analysis of model performance at lower costs. Polyrating can detect and quantify biases affecting human preferences, ensuring fairer model comparisons. Further, Polyrating can reduce the cost of human evaluations by up to $41\%$ for new models and up to $77\%$ for new tasks by leveraging existing benchmark scores. Lastly, Polyrating enables direct comparisons of ratings across different tasks, providing a comprehensive understanding of an LLMs' strengths, weaknesses, and relative performance across different applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04411v2">Mediator: Memory-efficient LLM Merging with Less Parameter Conflicts and Uncertainty Based Routing</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 work in progress. arXiv admin note: text overlap with arXiv:2405.09673 by other authors
    </div>
    <details class="paper-abstract">
      Model merging aggregates Large Language Models (LLMs) finetuned on different tasks into a stronger one. However, parameter conflicts between models leads to performance degradation in averaging. While model routing addresses this issue by selecting individual models during inference, it imposes excessive storage and compute costs, and fails to leverage the common knowledge from different models. In this work, we observe that different layers exhibit varying levels of parameter conflicts. Building on this insight, we average layers with minimal parameter conflicts and use a novel task-level expert routing for layers with significant conflicts. To further reduce storage costs, inspired by task arithmetic sparsity, we decouple multiple fine-tuned experts into a dense expert and several sparse experts. Considering the out-of-distribution samples, we select and merge appropriate experts based on the task uncertainty of the input data. We conduct extensive experiments on both LLaMA and Qwen with varying parameter scales, and evaluate on real-world reasoning tasks. Results demonstrate that our method consistently achieves significant performance improvements while requiring less system cost compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07495v1">LLM-Sketch: Enhancing Network Sketches with LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Network stream mining is fundamental to many network operations. Sketches, as compact data structures that offer low memory overhead with bounded accuracy, have emerged as a promising solution for network stream mining. Recent studies attempt to optimize sketches using machine learning; however, these approaches face the challenges of lacking adaptivity to dynamic networks and incurring high training costs. In this paper, we propose LLM-Sketch, based on the insight that fields beyond the flow IDs in packet headers can also help infer flow sizes. By using a two-tier data structure and separately recording large and small flows, LLM-Sketch improves accuracy while minimizing memory usage. Furthermore, it leverages fine-tuned large language models (LLMs) to reliably estimate flow sizes. We evaluate LLM-Sketch on three representative tasks, and the results demonstrate that LLM-Sketch outperforms state-of-the-art methods by achieving a $7.5\times$ accuracy improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13341v2">Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 ICLR 2025; 28 pages, 8 figures
    </div>
    <details class="paper-abstract">
      High quality annotations are increasingly a bottleneck in the explosively growing machine learning ecosystem. Scalable evaluation methods that avoid costly annotation have therefore become an important research ambition. Many hope to use strong existing models in lieu of costly labels to provide cheap model evaluations. Unfortunately, this method of using models as judges introduces biases, such as self-preferencing, that can distort model comparisons. An emerging family of debiasing tools promises to fix these issues by using a few high quality labels to debias a large number of model judgments. In this paper, we study how far such debiasing methods, in principle, can go. Our main result shows that when the judge is no more accurate than the evaluated model, no debiasing method can decrease the required amount of ground truth labels by more than half. Our result speaks to the severe limitations of the LLM-as-a-judge paradigm at the evaluation frontier where the goal is to assess newly released models that are possibly better than the judge. Through an empirical evaluation, we demonstrate that the sample size savings achievable in practice are even more modest than what our theoretical limit suggests. Along the way, our work provides new observations about debiasing methods for model evaluation, and points out promising avenues for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07459v1">PerCul: A Story-Driven Cultural Evaluation of LLMs in Persian</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Accepted at NAACL 2025 Main Conference, the dataset is available on HuggingFace (see https://huggingface.co/datasets/teias-ai/percul)
    </div>
    <details class="paper-abstract">
      Large language models predominantly reflect Western cultures, largely due to the dominance of English-centric training data. This imbalance presents a significant challenge, as LLMs are increasingly used across diverse contexts without adequate evaluation of their cultural competence in non-English languages, including Persian. To address this gap, we introduce PerCul, a carefully constructed dataset designed to assess the sensitivity of LLMs toward Persian culture. PerCul features story-based, multiple-choice questions that capture culturally nuanced scenarios. Unlike existing benchmarks, PerCul is curated with input from native Persian annotators to ensure authenticity and to prevent the use of translation as a shortcut. We evaluate several state-of-the-art multilingual and Persian-specific LLMs, establishing a foundation for future research in cross-cultural NLP evaluation. Our experiments demonstrate a 11.3% gap between best closed source model and layperson baseline while the gap increases to 21.3% by using the best open-weight model. You can access the dataset from here: https://huggingface.co/datasets/teias-ai/percul
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07445v1">Forget What You Know about LLMs Evaluations - LLMs are Like a Chameleon</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often appear to excel on public benchmarks, but these high scores may mask an overreliance on dataset-specific surface cues rather than true language understanding. We introduce the Chameleon Benchmark Overfit Detector (C-BOD), a meta-evaluation framework that systematically distorts benchmark prompts via a parametric transformation and detects overfitting of LLMs. By rephrasing inputs while preserving their semantic content and labels, C-BOD exposes whether a model's performance is driven by memorized patterns. Evaluated on the MMLU benchmark using 26 leading LLMs, our method reveals an average performance degradation of 2.15% under modest perturbations, with 20 out of 26 models exhibiting statistically significant differences. Notably, models with higher baseline accuracy exhibit larger performance differences under perturbation, and larger LLMs tend to be more sensitive to rephrasings indicating that both cases may overrely on fixed prompt patterns. In contrast, the Llama family and models with lower baseline accuracy show insignificant degradation, suggesting reduced dependency on superficial cues. Moreover, C-BOD's dataset- and model-agnostic design allows easy integration into training pipelines to promote more robust language understanding. Our findings challenge the community to look beyond leaderboard scores and prioritize resilience and generalization in LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07443v1">Approximating Human Strategic Reasoning with LLM-Enhanced Recursive Reasoners Leveraging Multi-agent Hypergames</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      LLM-driven multi-agent-based simulations have been gaining traction with applications in game-theoretic and social simulations. While most implementations seek to exploit or evaluate LLM-agentic reasoning, they often do so with a weak notion of agency and simplified architectures. We implement a role-based multi-agent strategic interaction framework tailored to sophisticated recursive reasoners, providing the means for systematic in-depth development and evaluation of strategic reasoning. Our game environment is governed by the umpire responsible for facilitating games, from matchmaking through move validation to environment management. Players incorporate state-of-the-art LLMs in their decision mechanism, relying on a formal hypergame-based model of hierarchical beliefs. We use one-shot, 2-player beauty contests to evaluate the recursive reasoning capabilities of the latest LLMs, providing a comparison to an established baseline model from economics and data from human experiments. Furthermore, we introduce the foundations of an alternative semantic measure of reasoning to the k-level theory. Our experiments show that artificial reasoners can outperform the baseline model in terms of both approximating human behaviour and reaching the optimal solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07424v1">RomanLens: Latent Romanization and its role in Multilinguality in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 18 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit remarkable multilingual generalization despite being predominantly trained on English-centric corpora. A fundamental question arises: how do LLMs achieve such robust multilingual capabilities? For non-Latin script languages, we investigate the role of romanization - the representation of non-Latin scripts using Latin characters - as a bridge in multilingual processing. Using mechanistic interpretability techniques, we analyze next-token generation and find that intermediate layers frequently represent target words in romanized form before transitioning to native script, a phenomenon we term Latent Romanization. Further, through activation patching experiments, we demonstrate that LLMs encode semantic concepts similarly across native and romanized scripts, suggesting a shared underlying representation. Additionally in translation towards non Latin languages, our findings reveal that when the target language is in romanized form, its representations emerge earlier in the model's layers compared to native script. These insights contribute to a deeper understanding of multilingual representation in LLMs and highlight the implicit role of romanization in facilitating language transfer. Our work provides new directions for potentially improving multilingual language modeling and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00560v2">Re-evaluating Automatic LLM System Ranking for Alignment with Human Preference</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Findings of NAACL 2025
    </div>
    <details class="paper-abstract">
      Evaluating and ranking the capabilities of different LLMs is crucial for understanding their performance and alignment with human preferences. Due to the high cost and time-consuming nature of human evaluations, an automatic LLM bencher (i.e., an automatic evaluation framework that aims to rank LLMs based on their alignment with human preferences) is indispensable. An automatic LLM bencher consists of four components: the input set (e.g., a user instruction), the evaluation model (e.g., an LLM), the evaluation type (e.g., pairwise comparison), and the aggregation method (e.g., the ELO rating system). However, previous work has not thoroughly explored how to select these components or how their different combinations influence the results. In this work, through controlled experiments, we provide a series of recommendations on how to choose each component to better automate the evaluation of LLMs. Furthermore, we discovered that when evaluating LLMs with similar performance, the performance of the automatic LLM bencher declines sharply, underscoring the limitations of current benchers and calling for future work. Lastly, we found that the evaluation models' performance at the instance level (e.g., the accuracy of selecting the best output) does not always align with their effectiveness when used as a component of a bencher, highlighting the importance of dedicated system-level evaluation of benchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07418v1">Entity Linking using LLMs for Automated Product Carbon Footprint Estimation</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Growing concerns about climate change and sustainability are driving manufacturers to take significant steps toward reducing their carbon footprints. For these manufacturers, a first step towards this goal is to identify the environmental impact of the individual components of their products. We propose a system leveraging large language models (LLMs) to automatically map components from manufacturer Bills of Materials (BOMs) to Life Cycle Assessment (LCA) database entries by using LLMs to expand on available component information. Our approach reduces the need for manual data processing, paving the way for more accessible sustainability practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07374v1">LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Large reasoning models (LRMs) tackle complex reasoning problems by following long chain-of-thoughts (Long CoT) that incorporate reflection, backtracking, and self-validation. However, the training techniques and data requirements to elicit Long CoT remain poorly understood. In this work, we find that a Large Language model (LLM) can effectively learn Long CoT reasoning through data-efficient supervised fine-tuning (SFT) and parameter-efficient low-rank adaptation (LoRA). With just 17k long CoT training samples, the Qwen2.5-32B-Instruct model achieves significant improvements on a wide range of math and coding benchmarks, including 56.7% (+40.0%) on AIME 2024 and 57.0% (+8.1%) on LiveCodeBench, competitive to the proprietary o1-preview model's score of 44.6% and 59.1%. More importantly, we find that the structure of Long CoT is critical to the learning process, whereas the content of individual reasoning steps has minimal impact. Perturbations affecting content, such as training on incorrect samples or removing reasoning keywords, have little impact on performance. In contrast, structural modifications that disrupt logical consistency in the Long CoT, such as shuffling or deleting reasoning steps, significantly degrade accuracy. For example, a model trained on Long CoT samples with incorrect answers still achieves only 3.2% lower accuracy compared to training with fully correct samples. These insights deepen our understanding of how to elicit reasoning capabilities in LLMs and highlight key considerations for efficiently training the next generation of reasoning models. This is the academic paper of our previous released Sky-T1-32B-Preview model. Codes are available at https://github.com/NovaSky-AI/SkyThought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12109v2">Can LLMs Learn Macroeconomic Narratives from Social Media?</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      This study empirically tests the $\textit{Narrative Economics}$ hypothesis, which posits that narratives (ideas that are spread virally and affect public beliefs) can influence economic fluctuations. We introduce two curated datasets containing posts from X (formerly Twitter) which capture economy-related narratives (Data will be shared upon paper acceptance). Employing Natural Language Processing (NLP) methods, we extract and summarize narratives from the tweets. We test their predictive power for $\textit{macroeconomic}$ forecasting by incorporating the tweets' or the extracted narratives' representations in downstream financial prediction tasks. Our work highlights the challenges in improving macroeconomic models with narrative data, paving the way for the research community to realistically address this important challenge. From a scientific perspective, our investigation offers valuable insights and NLP tools for narrative extraction and summarization using Large Language Models (LLMs), contributing to future research on the role of narratives in economics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06351v2">Calibrating LLMs with Information-Theoretic Evidential Deep Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 27 pages; 3 figures; accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Fine-tuned large language models (LLMs) often exhibit overconfidence, particularly when trained on small datasets, resulting in poor calibration and inaccurate uncertainty estimates. Evidential Deep Learning (EDL), an uncertainty-aware approach, enables uncertainty estimation in a single forward pass, making it a promising method for calibrating fine-tuned LLMs. However, despite its computational efficiency, EDL is prone to overfitting, as its training objective can result in overly concentrated probability distributions. To mitigate this, we propose regularizing EDL by incorporating an information bottleneck (IB). Our approach IB-EDL suppresses spurious information in the evidence generated by the model and encourages truly predictive information to influence both the predictions and uncertainty estimates. Extensive experiments across various fine-tuned LLMs and tasks demonstrate that IB-EDL outperforms both existing EDL and non-EDL approaches. By improving the trustworthiness of LLMs, IB-EDL facilitates their broader adoption in domains requiring high levels of confidence calibration. Code is available at https://github.com/sandylaker/ib-edl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11507v3">Revisiting Benchmark and Assessment: An Agent-based Exploratory Dynamic Evaluation Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      While various vertical domain large language models (LLMs) have been developed, automatically evaluating their performance across different domains remains a critical challenge. Current benchmark-based methods often rely on static and costly datasets, are misaligned with practical user needs, and lack flexibility across domains. To address these limitations, we revisit the evaluation process and introduce two key concepts: Benchmark+, which extends the traditional question-answer benchmark into a more flexible ``strategy-criterion'' format; and Assessment+, which enhances the interaction process, enabling deeper exploration and supporting analysis from broader perspectives. We propose TestAgent, an agent-based evaluation framework that implements these concepts using retrieval-augmented generation and reinforcement learning. TestAgent enables automatic dynamic benchmark generation and in-depth assessment across diverse vertical domain scenarios. Experiments on tasks ranging from constructing multiple vertical domain evaluations to converting static benchmarks into dynamic forms demonstrate the effectiveness of TestAgent. This work offers an interesting perspective on automatic evaluation for LLMs and highlights a pathway for dynamic and domain-adaptive assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04620v5">CataractBot: An LLM-Powered Expert-in-the-Loop Chatbot for Cataract Patients</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      The healthcare landscape is evolving, with patients seeking reliable information about their health conditions and available treatment options. Despite the abundance of information sources, the digital age overwhelms individuals with excess, often inaccurate information. Patients primarily trust medical professionals, highlighting the need for expert-endorsed health information. However, increased patient loads on experts has led to reduced communication time, impacting information sharing. To address this gap, we developed CataractBot. CataractBot answers cataract surgery related questions instantly using an LLM to query a curated knowledge base, and provides expert-verified responses asynchronously. It has multimodal and multilingual capabilities. In an in-the-wild deployment study with 49 patients and attendants, 4 doctors, and 2 patient coordinators, CataractBot demonstrated potential, providing anytime accessibility, saving time, accommodating diverse literacy levels, alleviating power differences, and adding a privacy layer between patients and doctors. Users reported that their trust in the system was established through expert verification. Broadly, our results could inform future work on designing expert-mediated LLM bots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07266v1">When More is Less: Understanding Chain-of-Thought Length in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) reasoning enhances the multi-step reasoning capabilities of large language models (LLMs) by breaking complex tasks into smaller, manageable sub-tasks. Researchers have been exploring ways to guide models to generate more complex CoT processes to improve the reasoning ability of LLMs, such as long CoT and the test-time scaling law. However, for most models and tasks, does an increase in CoT length consistently lead to improved reasoning accuracy? In this paper, we observe a nuanced relationship: as the number of reasoning steps increases, performance initially improves but eventually decreases. To understand this phenomenon, we provide a piece of evidence that longer reasoning processes are increasingly susceptible to noise. We theoretically prove the existence of an optimal CoT length and derive a scaling law for this optimal length based on model capability and task difficulty. Inspired by our theory, we conduct experiments on both synthetic and real world datasets and propose Length-filtered Vote to alleviate the effects of excessively long or short CoTs. Our findings highlight the critical need to calibrate CoT length to align with model capabilities and task demands, offering a principled framework for optimizing multi-step reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00290v2">Estimating LLM Uncertainty with Logits</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Fixed some data errors in Table 1
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have seen remarkable advancements and have been extensively integrated across various fields. Despite their progress, LLMs are prone to hallucinations, producing responses that may not be dependable if the models lack sufficient grounding knowledge. To mitigate this issue, methods for estimating uncertainty have been adopted, with a focus on critical tokens as indicators of reliability. Nevertheless, probability-based approaches have shown limitations in assessing token-level reliability due to the erosion of evidence strength information acquired during training. In this paper, we introduce Logits-induced Token Uncertainty (LogU), a novel framework designed to estimate token-specific uncertainty in LLMs in real time, without the need for multiple sampling rounds. By leveraging evidence modeling for the implementation of LogU, we utilize the derived uncertainty measures to steer downstream tasks. Our experimental findings highlight the substantial effectiveness and potential of LogU, marking a significant advancement in addressing the challenge of model hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00212v3">STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 23 pages, 5 figures
    </div>
    <details class="paper-abstract">
      A fundamental challenge in formal theorem proving by LLMs is the lack of high-quality training data. Although reinforcement learning or expert iteration partially mitigates this issue by alternating between LLM generating proofs and finetuning them on correctly generated ones, performance quickly plateaus due to the scarcity of correct proofs (sparse rewards). To keep improving the models with limited data, we draw inspiration from mathematicians, who continuously develop new results, partly by proposing novel conjectures or exercises (which are often variants of known results) and attempting to solve them. We design the Self-play Theorem Prover (STP) that simultaneously takes on two roles, conjecturer and prover, each providing training signals to the other. The conjecturer is trained iteratively on previously generated conjectures that are barely provable by the current prover, which incentivizes it to generate increasingly challenging conjectures over time. The prover attempts to prove the conjectures with standard expert iteration. We evaluate STP with both Lean and Isabelle formal versifiers. With 19.8 billion tokens generated during the training in Lean, STP proves 26.3% of the statements in the LeanWorkbook dataset, doubling the previous best result of 13.2% achieved through expert iteration. The final model achieves state-of-the-art performance among whole-proof generation methods on miniF2F-test (61.7%, pass@3200), Proofnet-test (23.1%, pass@3200) and PutnamBench (8/644, pass@3200).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06348v2">AiRacleX: Automated Detection of Price Oracle Manipulations via LLM-Driven Knowledge Mining and Prompt Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Decentralized finance (DeFi) applications depend on accurate price oracles to ensure secure transactions, yet these oracles are highly vulnerable to manipulation, enabling attackers to exploit smart contract vulnerabilities for unfair asset valuation and financial gain. Detecting such manipulations traditionally relies on the manual effort of experienced experts, presenting significant challenges. In this paper, we propose a novel LLM-driven framework that automates the detection of price oracle manipulations by leveraging the complementary strengths of different LLM models (LLMs). Our approach begins with domain-specific knowledge extraction, where an LLM model synthesizes precise insights about price oracle vulnerabilities from top-tier academic papers, eliminating the need for profound expertise from developers or auditors. This knowledge forms the foundation for a second LLM model to generate structured, context-aware chain of thought prompts, which guide a third LLM model in accurately identifying manipulation patterns in smart contracts. We validate the effectiveness of framework through experiments on 60 known vulnerabilities from 46 real-world DeFi attacks or projects spanning 2021 to 2023. The best performing combination of LLMs (Haiku-Haiku-4o-mini) identified by AiRacleX demonstrate a 2.58-times improvement in recall (0.667 vs 0.259) compared to the state-of-the-art tool GPTScan, while maintaining comparable precision. Furthermore, our framework demonstrates the feasibility of replacing commercial models with open-source alternatives, enhancing privacy and security for developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07218v1">LUNAR: LLM Unlearning via Neural Activation Redirection</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leaking private information. The ability to selectively remove knowledge from LLMs is, therefore, a highly desirable capability. In this paper, we propose LUNAR, a novel unlearning methodology grounded in the Linear Representation Hypothesis. LUNAR operates by redirecting the representations of unlearned data to regions that trigger the model's inherent ability to express its inability to answer. LUNAR achieves state-of-the-art unlearning performance while significantly enhancing the controllability of the unlearned model during inference. Specifically, LUNAR achieves between 2.9x to 11.7x improvements on combined "unlearning efficacy" and "model utility" score ("Deviation Score") on the PISTOL dataset across various base models. We also demonstrate, through quantitative analysis and qualitative examples, LUNAR's superior controllability in generating coherent and contextually aware responses, mitigating undesired side effects of existing methods. Moreover, we demonstrate that LUNAR is robust against white-box adversarial attacks and versatile in handling real-world scenarios, such as processing sequential unlearning requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13276v3">SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Attention is the cornerstone of modern Large Language Models (LLMs). Yet its quadratic complexity hinders efficiency and scalability, especially for long-context processing. A promising approach is to leverage sparsity in attention. However, existing sparsity-based solutions predominantly rely on predefined patterns or heuristics at the attention head level, struggling to adapt dynamically to different contexts efficiently. We propose SeerAttention, a simple yet effective attention mechanism that directly learns the block-level attention sparsity from the LLM itself. Inspired by the gating mechanism in Mixture of Experts (MoE), SeerAttention augments the conventional attention with a learnable gate that selectively activates important blocks within the attention map. Specifically, the gate first pools the query (Q) and key (K) tensors along the sequence dimension and processes them through learnable linear layers. The resulting matrices are then multiplied together to produce the gating scores, which are used to predict block-level attention sparsity. Combined with our block-sparse FlashAttention kernel, SeerAttention can achieve significant speedup on GPUs. When applied to pre-trained LLMs, SeerAttention only requires training the gate parameters in a lightweight self-distillation manner, allowing rapid convergence. Our evaluation results demonstrate that SeerAttention achieves better model accuracy and lower latency for long-context pre-filling compared to prior methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07191v1">Bag of Tricks for Inference-time Computation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      With the advancement of large language models (LLMs), solving complex reasoning tasks has gained increasing attention. Inference-time computation methods (e.g., Best-of-N, beam search, et al.) are particularly valuable as they can enhance reasoning performance without modifying model parameters or requiring additional training. However, these techniques come with implementation challenges, and most existing methods remain at the proof-of-concept stage with limited practical adoption due to their computational complexity and varying effectiveness across different tasks. In this paper, we investigate and benchmark diverse inference-time computation strategies across reasoning tasks of varying complexity. Since most current methods rely on a proposer-verifier pipeline that first generates candidate solutions (e.g., reasoning solutions) and then selects the best one based on reward signals (e.g., RLHF rewards, process rewards), our research focuses on optimizing both candidate solution generation (e.g., instructing prompts, hyperparameters such as temperature and top-p) and reward mechanisms (e.g., self-evaluation, reward types). Through extensive experiments (more than 20,000 A100-80G GPU hours with over 1,000 experiments) across a variety of models (e.g., Llama, Qwen, and Mistral families) of various sizes, our ablation studies reveal that previously overlooked strategies can significantly enhance performance (e.g., tuning temperature can improve reasoning task performance by up to 5%). Furthermore, we establish a standardized benchmark for inference-time computation by systematically evaluating six representative methods across eight reasoning tasks. These findings provide a stronger foundation for future research. The code is available at https://github.com/usail-hkust/benchmark_inference_time_computation_LL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07190v1">Understanding LLMs' Fluid Intelligence Deficiency: An Analysis of the ARC Task</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 22 pages, 9 figures, accepted by NAACL 2025 main conference
    </div>
    <details class="paper-abstract">
      While LLMs have exhibited strong performance on various NLP tasks, it is noteworthy that most of these tasks rely on utilizing the vast amount of knowledge encoded in LLMs' parameters, rather than solving new problems without prior knowledge. In cognitive research, the latter ability is referred to as fluid intelligence, which is considered to be critical for assessing human intelligence. Recent research on fluid intelligence assessments has highlighted significant deficiencies in LLMs' abilities. In this paper, we analyze the challenges LLMs face in demonstrating fluid intelligence through controlled experiments, using the most representative ARC task as an example. Our study revealed three major limitations in existing LLMs: limited ability for skill composition, unfamiliarity with abstract input formats, and the intrinsic deficiency of left-to-right decoding. Our data and code can be found in https://wujunjie1998.github.io/araoc-benchmark.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07186v1">Perceived Confidence Scoring for Data Annotation with Zero-Shot LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Zero-shot LLMs are now also used for textual classification tasks, e.g., sentiment/emotion detection of a given input as a sentence/article. However, their performance can be suboptimal in such data annotation tasks. We introduce a novel technique Perceived Confidence Scoring (PCS) that evaluates LLM's confidence for its classification of an input by leveraging Metamorphic Relations (MRs). The MRs generate semantically equivalent yet textually mutated versions of the input. Following the principles of Metamorphic Testing (MT), the mutated versions are expected to have annotation labels similar to the input. By analyzing the consistency of LLM responses across these variations, PCS computes a confidence score based on the frequency of predicted labels. PCS can be used both for single LLM and multiple LLM settings (e.g., majority voting). We introduce an algorithm Perceived Differential Evolution (PDE) that determines the optimal weights assigned to the MRs and the LLMs for a classification task. Empirical evaluation shows PCS significantly improves zero-shot accuracy for Llama-3-8B-Instruct (4.96%) and Mistral-7B-Instruct-v0.3 (10.52%), with Gemma-2-9b-it showing a 9.39% gain. When combining all three models, PCS significantly outperforms majority voting by 7.75%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12665v3">CollabStory: Multi-LLM Collaborative Story Generation and Authorship Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Accepted to NAACL Findings 2025
    </div>
    <details class="paper-abstract">
      The rise of unifying frameworks that enable seamless interoperability of Large Language Models (LLMs) has made LLM-LLM collaboration for open-ended tasks a possibility. Despite this, there have not been efforts to explore such collaborative writing. We take the next step beyond human-LLM collaboration to explore this multi-LLM scenario by generating the first exclusively LLM-generated collaborative stories dataset called CollabStory. We focus on single-author to multi-author (up to 5 LLMs) scenarios, where multiple LLMs co-author stories. We generate over 32k stories using open-source instruction-tuned LLMs. Further, we take inspiration from the PAN tasks that have set the standard for human-human multi-author writing tasks and analysis. We extend their authorship-related tasks for multi-LLM settings and present baselines for LLM-LLM collaboration. We find that current baselines are not able to handle this emerging scenario. Thus, CollabStory is a resource that could help propel an understanding as well as the development of new techniques to discern the use of multiple LLMs. This is crucial to study in the context of writing tasks since LLM-LLM collaboration could potentially overwhelm ongoing challenges related to plagiarism detection, credit assignment, maintaining academic integrity in educational settings, and addressing copyright infringement concerns. We make our dataset and code available at https://github.com/saranya-venkatraman/CollabStory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04564v2">My LLM might Mimic AAE -- But When Should it?</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      We examine the representation of African American English (AAE) in large language models (LLMs), exploring (a) the perceptions Black Americans have of how effective these technologies are at producing authentic AAE, and (b) in what contexts Black Americans find this desirable. Through both a survey of Black Americans ($n=$ 104) and annotation of LLM-produced AAE by Black Americans ($n=$ 228), we find that Black Americans favor choice and autonomy in determining when AAE is appropriate in LLM output. They tend to prefer that LLMs default to communicating in Mainstream U.S. English in formal settings, with greater interest in AAE production in less formal settings. When LLMs were appropriately prompted and provided in context examples, our participants found their outputs to have a level of AAE authenticity on par with transcripts of Black American speech. Select code and data for our project can be found here: https://github.com/smelliecat/AAEMime.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18794v2">Open-Nav: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environment with Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Accepted by ICRA 2025
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) tasks require an agent to follow textual instructions to navigate through 3D environments. Traditional approaches use supervised learning methods, relying heavily on domain-specific datasets to train VLN models. Recent methods try to utilize closed-source large language models (LLMs) like GPT-4 to solve VLN tasks in zero-shot manners, but face challenges related to expensive token costs and potential data breaches in real-world applications. In this work, we introduce Open-Nav, a novel study that explores open-source LLMs for zero-shot VLN in the continuous environment. Open-Nav employs a spatial-temporal chain-of-thought (CoT) reasoning approach to break down tasks into instruction comprehension, progress estimation, and decision-making. It enhances scene perceptions with fine-grained object and spatial knowledge to improve LLM's reasoning in navigation. Our extensive experiments in both simulated and real-world environments demonstrate that Open-Nav achieves competitive performance compared to using closed-source LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07143v1">Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Accurate and efficient diagnosis in online medical consultations remains a challenge for current large language models. These models often rely on single-turn interactions and lack the ability to refine their predictions through follow-up questions. Additionally, their responses frequently contain complex medical terminology, making them less accessible to non-medical users and creating barriers to effective communication. In this paper, we introduce Ask Patients with Patience (APP), the first multi-turn dialogue that enables LLMs to iteratively refine diagnoses based on grounded reasoning. By integrating medical guidelines and entropy minimization, APP improves both diagnostic accuracy and efficiency. Furthermore, it features human-centric communication that bridges the gap between user comprehension and medical terminology, significantly enhancing user accessibility and engagement. We evaluated APP using a subset of the ReMeDi dataset, comparing it with single-turn and traditional multi-turn LLM baselines. APP achieved higher similarity scores in diagnosis predictions, demonstrating better alignment with ground truth diagnoses. Entropy analysis showed that APP reduces diagnostic uncertainty more rapidly across iterations, increasing confidence in its predictions. APP also excels in user accessibility and empathy, further bridging the gap between complex medical language and user understanding. Code will be released at: https://github.com/SuperMedIntel/AskPatients.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01618v3">A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved significant performance gains via scaling up model sizes and/or data. However, recent evidence suggests diminishing returns from such approaches, motivating scaling the computation spent at inference time. Existing inference-time scaling methods, usually with reward models, cast the task as a search problem, which tends to be vulnerable to reward hacking as a consequence of approximation errors in reward models. In this paper, we instead cast inference-time scaling as a probabilistic inference task and leverage sampling-based techniques to explore the typical set of the state distribution of a state-space model with an approximate likelihood, rather than optimize for its mode directly. We propose a novel inference-time scaling approach by adapting particle-based Monte Carlo methods to this task. Our empirical evaluation demonstrates that our methods have a 4-16x better scaling rate over our deterministic search counterparts on various challenging mathematical reasoning tasks. Using our approach, we show that Qwen2.5-Math-1.5B-Instruct can surpass GPT-4o accuracy in only 4 rollouts, while Qwen2.5-Math-7B-Instruct scales to o1 level accuracy in only 32 rollouts. Our work not only presents an effective method to inference-time scaling, but also connects the rich literature in probabilistic inference with inference-time scaling of LLMs to develop more robust algorithms in future work. Code, videos, and further information available at https://probabilistic-inference-scaling.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.24155v2">Blind Spot Navigation in LLM Reasoning with Thought Space Explorer</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated their potential in handling complex reasoning tasks, which are usually achieved by constructing a thought chain to guide the model to solve the problem with multi-step thinking. However, existing methods often remain confined to previously explored solution spaces and thus overlook the critical blind spot within LLMs' cognitive range. To address these issues, we design the Thought Space Explorer (TSE), a novel framework to expand and optimize thought structures to guide LLMs to explore their blind spots of thinking. By generating new reasoning steps and branches based on the original thought structure with various designed strategies, TSE broadens the thought space and alleviates the impact of blind spots for LLM reasoning. Experimental results on multiple levels of reasoning tasks demonstrate the efficacy of TSE. We also conduct extensive analysis to understand how structured and expansive thought can contribute to unleashing the potential of LLM reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07987v1">Universal Adversarial Attack on Aligned Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., ''Sure, here it is'') or otherwise unsafe content-even for harmful prompts. In experiments on the SafeBench benchmark, our method achieves significantly higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 93% on certain models). We further demonstrate cross-model transferability by training on several multimodal LLMs simultaneously and testing on unseen architectures. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive to some readers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07982v1">Deep Semantic Graph Learning via LLM based Node Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Graph learning has attracted significant attention due to its widespread real-world applications. Current mainstream approaches rely on text node features and obtain initial node embeddings through shallow embedding learning using GNNs, which shows limitations in capturing deep textual semantics. Recent advances in Large Language Models (LLMs) have demonstrated superior capabilities in understanding text semantics, transforming traditional text feature processing. This paper proposes a novel framework that combines Graph Transformer architecture with LLM-enhanced node features. Specifically, we leverage LLMs to generate rich semantic representations of text nodes, which are then processed by a multi-head self-attention mechanism in the Graph Transformer to capture both local and global graph structural information. Our model utilizes the Transformer's attention mechanism to dynamically aggregate neighborhood information while preserving the semantic richness provided by LLM embeddings. Experimental results demonstrate that the LLM-enhanced node features significantly improve the performance of graph learning models on node classification tasks. This approach shows promising results across multiple graph learning tasks, offering a practical direction for combining graph networks with language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07980v1">CIRCUIT: A Benchmark for Circuit Interpretation and Reasoning Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      The role of Large Language Models (LLMs) has not been extensively explored in analog circuit design, which could benefit from a reasoning-based approach that transcends traditional optimization techniques. In particular, despite their growing relevance, there are no benchmarks to assess LLMs' reasoning capability about circuits. Therefore, we created the CIRCUIT dataset consisting of 510 question-answer pairs spanning various levels of analog-circuit-related subjects. The best-performing model on our dataset, GPT-4o, achieves 48.04% accuracy when evaluated on the final numerical answer. To evaluate the robustness of LLMs on our dataset, we introduced a unique feature that enables unit-test-like evaluation by grouping questions into unit tests. In this case, GPT-4o can only pass 27.45% of the unit tests, highlighting that the most advanced LLMs still struggle with understanding circuits, which requires multi-level reasoning, particularly when involving circuit topologies. This circuit-specific benchmark highlights LLMs' limitations, offering valuable insights for advancing their application in analog integrated circuit design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07974v1">From Hazard Identification to Controller Design: Proactive and LLM-Supported Safety Engineering for ML-Powered Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Accepted for publication at the International Conference on AI Engineering (CAIN) 2025
    </div>
    <details class="paper-abstract">
      Machine learning (ML) components are increasingly integrated into software products, yet their complexity and inherent uncertainty often lead to unintended and hazardous consequences, both for individuals and society at large. Despite these risks, practitioners seldom adopt proactive approaches to anticipate and mitigate hazards before they occur. Traditional safety engineering approaches, such as Failure Mode and Effects Analysis (FMEA) and System Theoretic Process Analysis (STPA), offer systematic frameworks for early risk identification but are rarely adopted. This position paper advocates for integrating hazard analysis into the development of any ML-powered software product and calls for greater support to make this process accessible to developers. By using large language models (LLMs) to partially automate a modified STPA process with human oversight at critical steps, we expect to address two key challenges: the heavy dependency on highly experienced safety engineering experts, and the time-consuming, labor-intensive nature of traditional hazard analysis, which often impedes its integration into real-world development workflows. We illustrate our approach with a running example, demonstrating that many seemingly unanticipated issues can, in fact, be anticipated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05400v2">Dynamic Noise Preference Optimization for LLM Self-Improvement via Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 Due to an update in the company's publication approval process, a newly appointed manager has been added to the review workflow. As a result, we need to resubmit the application for approval under the revised process. Therefore, we are temporarily withdrawing this submission until the new approval workflow is completed
    </div>
    <details class="paper-abstract">
      Although LLMs have achieved significant success, their reliance on large volumes of human-annotated data has limited their potential for further scaling. In this situation, utilizing self-generated synthetic data has become crucial for fine-tuning LLMs without extensive human annotation. However, current methods often fail to ensure consistent improvements across iterations, with performance stagnating after only minimal updates. To overcome these challenges, we introduce Dynamic Noise Preference Optimization (DNPO). DNPO employs a dynamic sample labeling mechanism to construct preference pairs for training and introduces controlled, trainable noise into the preference optimization process. Our approach effectively prevents stagnation and enables continuous improvement. In experiments with Zephyr-7B, DNPO consistently outperforms existing methods, showing an average performance boost of 2.6% across multiple benchmarks. Additionally, DNPO shows a significant improvement in model-generated data quality, with a 29.4% win-loss rate gap compared to the baseline in GPT-4 evaluations. This highlights its effectiveness in enhancing model performance through iterative refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07963v1">Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature?</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 20 pages, 10 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Medical research faces well-documented challenges in translating novel treatments into clinical practice. Publishing incentives encourage researchers to present "positive" findings, even when empirical results are equivocal. Consequently, it is well-documented that authors often spin study results, especially in article abstracts. Such spin can influence clinician interpretation of evidence and may affect patient care decisions. In this study, we ask whether the interpretation of trial results offered by Large Language Models (LLMs) is similarly affected by spin. This is important since LLMs are increasingly being used to trawl through and synthesize published medical evidence. We evaluated 22 LLMs and found that they are across the board more susceptible to spin than humans. They might also propagate spin into their outputs: We find evidence, e.g., that LLMs implicitly incorporate spin into plain language summaries that they generate. We also find, however, that LLMs are generally capable of recognizing spin, and can be prompted in a way to mitigate spin's impact on LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07942v1">Symbiotic Cooperation for Web Agents: Harnessing Complementary Strengths of Large and Small LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Web browsing agents powered by large language models (LLMs) have shown tremendous potential in automating complex web-based tasks. Existing approaches typically rely on large LLMs (e.g., GPT-4o) to explore web environments and generate trajectory data, which is then used either for demonstration retrieval (for large LLMs) or to distill small LLMs (e.g., Llama3) in a process that remains decoupled from the exploration. In this paper, we propose AgentSymbiotic, an iterative framework that couples data synthesis with task-performance, yielding a "symbiotic improvement" for both large and small LLMs. Our study uncovers a complementary dynamic between LLM types: while large LLMs excel at generating high-quality trajectories for distillation, the distilled small LLMs-owing to their distinct reasoning capabilities-often choose actions that diverge from those of their larger counterparts. This divergence drives the exploration of novel trajectories, thereby enriching the synthesized data. However, we also observe that the performance of small LLMs becomes a bottleneck in this iterative enhancement process. To address this, we propose two innovations in LLM distillation: a speculative data synthesis strategy that mitigates off-policy bias, and a multi-task learning approach designed to boost the reasoning capabilities of the student LLM. Furthermore, we introduce a Hybrid Mode for Privacy Preservation to address user privacy concerns. Evaluated on the WEBARENA benchmark, AgentSymbiotic achieves SOTA performance with both LLM types. Our best Large LLM agent reaches 52%, surpassing the previous best of 45%, while our 8B distilled model demonstrates a competitive 49%, exceeding the prior best of 28%. Code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06111v2">CSR-Bench: Benchmarking LLM Agents in Deployment of Computer Science Research Repositories</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      The increasing complexity of computer science research projects demands more effective tools for deploying code repositories. Large Language Models (LLMs), such as Anthropic Claude and Meta Llama, have demonstrated significant advancements across various fields of computer science research, including the automation of diverse software engineering tasks. To evaluate the effectiveness of LLMs in handling complex code development tasks of research projects, particularly for NLP/CV/AI/ML/DM topics, we introduce CSR-Bench, a benchmark for Computer Science Research projects. This benchmark assesses LLMs from various aspects including accuracy, efficiency, and deployment script quality, aiming to explore their potential in conducting computer science research autonomously. We also introduce a novel framework, CSR-Agents, that utilizes multiple LLM agents to automate the deployment of GitHub code repositories of computer science research projects. Specifically, by checking instructions from markdown files and interpreting repository structures, the model generates and iteratively improves bash commands that set up the experimental environments and deploy the code to conduct research tasks. Preliminary results from CSR-Bench indicate that LLM agents can significantly enhance the workflow of repository deployment, thereby boosting developer productivity and improving the management of developmental workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07928v1">Distributed Approach to Haskell Based Applications Refactoring with LLMs Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      We present a large language models (LLMs) based multi-agent system to automate the refactoring of Haskell codebases. The multi-agent system consists of specialized agents performing tasks such as context analysis, refactoring, validation, and testing. Refactoring improvements are using metrics such as cyclomatic complexity, run-time, and memory allocation. Experimental evaluations conducted on Haskell codebases demonstrate improvements in code quality. Cyclomatic complexity was reduced by 13.64% and 47.06% in the respective codebases. Memory allocation improved by 4.17% and 41.73%, while runtime efficiency increased by up to 50%. These metrics highlight the systems ability to optimize Haskells functional paradigms while maintaining correctness and scalability. Results show reductions in complexity and performance enhancements across codebases. The integration of LLMs based multi-agent system enables precise task execution and inter-agent collaboration, addressing the challenges of refactoring in functional programming. This approach aims to address the challenges of refactoring functional programming languages through distributed and modular systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11876v3">Rescriber: Smaller-LLM-Powered User-Led Data Minimization for LLM-Based Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      The proliferation of LLM-based conversational agents has resulted in excessive disclosure of identifiable or sensitive information. However, existing technologies fail to offer perceptible control or account for users' personal preferences about privacy-utility tradeoffs due to the lack of user involvement. To bridge this gap, we designed, built, and evaluated Rescriber, a browser extension that supports user-led data minimization in LLM-based conversational agents by helping users detect and sanitize personal information in their prompts. Our studies (N=12) showed that Rescriber helped users reduce unnecessary disclosure and addressed their privacy concerns. Users' subjective perceptions of the system powered by Llama3-8B were on par with that by GPT-4o. The comprehensiveness and consistency of the detection and sanitization emerge as essential factors that affect users' trust and perceived protection. Our findings confirm the viability of smaller-LLM-powered, user-facing, on-device privacy controls, presenting a promising approach to address the privacy and trust challenges of AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07912v1">Elevating Legal LLM Responses: Harnessing Trainable Logical Structures and Semantic Knowledge with Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved impressive results across numerous domains, yet they experience notable deficiencies in legal question-answering tasks. LLMs often generate generalized responses that lack the logical specificity required for expert legal advice and are prone to hallucination, providing answers that appear correct but are unreliable. Retrieval-Augmented Generation (RAG) techniques offer partial solutions to address this challenge, but existing approaches typically focus only on semantic similarity, neglecting the logical structure essential to legal reasoning. In this paper, we propose the Logical-Semantic Integration Model (LSIM), a novel supervised framework that bridges semantic and logical coherence. LSIM comprises three components: reinforcement learning predicts a structured fact-rule chain for each question, a trainable Deep Structured Semantic Model (DSSM) retrieves the most relevant candidate questions by integrating semantic and logical features, and in-context learning generates the final answer using the retrieved content. Our experiments on a real-world legal QA dataset-validated through both automated metrics and human evaluation-demonstrate that LSIM significantly enhances accuracy and reliability compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07903v1">HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment</a></div>
    <div class="paper-meta">
      📅 2025-02-11
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Disaggregating the prefill and decoding phases represents an effective new paradigm for generative inference of large language models (LLM), which eliminates prefill-decoding interference and optimizes resource allocation. However, it is still an open problem about how to deploy the disaggregated inference paradigm across a group of heterogeneous GPUs, which can be an economical alternative to deployment over homogeneous high-performance GPUs. Towards this end, we introduce HexGen-2, a distributed system for efficient and economical LLM serving on heterogeneous GPUs following the disaggregated paradigm. Built on top of HexGen, the core component of HexGen-2 is a scheduling algorithm that formalizes the allocation of disaggregated LLM inference computations and communications over heterogeneous GPUs and network connections as a constraint optimization problem. We leverage the graph partitioning and max-flow algorithms to co-optimize resource allocation, parallel strategies for distinct inference phases, and the efficiency of inter-phase key-value (KV) cache communications. We conduct extensive experiments to evaluate HexGen-2, i.e., on OPT (30B) and Llama-2 (70B) models in various real-world settings, the results reveal that HexGen-2 delivers up to a 2.0 times and on average a 1.3 times improvement in serving throughput, reduces the average inference latency by 1.5 times compared with state-of-the-art systems given the same price budget, and achieves comparable inference performance with a 30% lower price budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07835v1">Bridging LLM-Generated Code and Requirements: Reverse Generation technique and SBC Metric for Developer Insights</a></div>
    <div class="paper-meta">
      📅 2025-02-11
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) in software engineering, particularly in code generation, has garnered significant attention. However, assessing the quality of AI-generated code remains a challenge due to the inherent complexity of programming tasks and the lack of robust evaluation metrics that align well with human judgment. Traditional token-based metrics such as BLEU and ROUGE, while commonly used in natural language processing, exhibit weak correlations with human assessments in code intelligence and verification tasks. Furthermore, these metrics are primarily research focused and are not designed for seamless integration into the software development lifecycle, limiting their practical utility for developers seeking to improve code quality and security. AI-assisted coding has been shown to be more beneficial for senior developers, as they possess the expertise to critically evaluate the generated code for correctness, completeness, and compliance. In contrast, junior developers may struggle to identify hallucinations, missing functionality, or incorrect logic in AI-generated code. To bridge this gap, This paper introduces a novel scoring mechanism called the SBC score, which is based on a reverse generation technique that leverages the natural language generation capabilities of LLMs. Unlike direct code analysis, our approach reconstructs system requirements from AI-generated code and compares them with the original specifications to quantify accuracy. The SBC score combines semantic similarity, BLEU, and completeness analysis, providing actionable insights to developers by highlighting missing features and hallucinations. Our code and datasets are available on GitHub
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06773v1">On the Emergence of Thinking in LLMs I: Searching for the Right Intuition</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 Abstract shortened for arXiv
    </div>
    <details class="paper-abstract">
      Recent AI advancements, such as OpenAI's new models, are transforming LLMs into LRMs (Large Reasoning Models) that perform reasoning during inference, taking extra time and compute for higher-quality outputs. We aim to uncover the algorithmic framework for training LRMs. Methods like self-consistency, PRM, and AlphaZero suggest reasoning as guided search. We ask: what is the simplest, most scalable way to enable search in LLMs? We propose a post-training framework called Reinforcement Learning via Self-Play (RLSP). RLSP involves three steps: (1) supervised fine-tuning with human or synthetic demonstrations of the reasoning process, (2) using an exploration reward signal to encourage diverse and efficient reasoning behaviors, and (3) RL training with an outcome verifier to ensure correctness while preventing reward hacking. Our key innovation is to decouple exploration and correctness signals during PPO training, carefully balancing them to improve performance and efficiency. Empirical studies in the math domain show that RLSP improves reasoning. On the Llama-3.1-8B-Instruct model, RLSP can boost performance by 23% in MATH-500 test set; On AIME 2024 math problems, Qwen2.5-32B-Instruct improved by 10% due to RLSP. However, a more important finding of this work is that the models trained using RLSP, even with the simplest exploration reward that encourages the model to take more intermediate steps, showed several emergent behaviors such as backtracking, exploration of ideas, and verification. These findings demonstrate that RLSP framework might be enough to enable emergence of complex reasoning abilities in LLMs when scaled. Lastly, we propose a theory as to why RLSP search strategy is more suitable for LLMs inspired by a remarkable result that says CoT provably increases computational power of LLMs, which grows as the number of steps in CoT \cite{li2024chain,merrill2023expresssive}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06772v1">ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 Code: https://github.com/Gen-Verse/ReasonFlux
    </div>
    <details class="paper-abstract">
      We present that hierarchical LLM reasoning via scaling thought templates can effectively optimize the reasoning search space and outperform the mathematical reasoning capabilities of powerful LLMs like OpenAI o1-preview and DeepSeek V3. We train our ReasonFlux-32B model with only 8 GPUs and introduces three innovations: (i) a structured and generic thought template library, containing around 500 high-level thought templates capable of generalizing to similar or relevant reasoning problems; (ii) performing hierarchical reinforcement learning on a sequence of thought templates instead of long CoTs, optimizing a base LLM to plan out an optimal template trajectory for gradually handling complex problems; (iii) a brand new inference scaling system that enables hierarchical LLM reasoning by adaptively scaling thought templates at inference time. With a template trajectory containing sequential thought templates, our ReasonFlux-32B significantly advances math reasoning capabilities to state-of-the-art levels. Notably, on the MATH benchmark, it achieves an accuracy of 91.2% and surpasses o1-preview by 6.7%. On the USA Math Olympiad (AIME) benchmark, ReasonFlux-32B solves an average of 56.7% of problems, surpassing o1-preview and DeepSeek-V3 by 27% and 45%, respectively. Code: https://github.com/Gen-Verse/ReasonFlux
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10626v2">Efficiently Democratizing Medical LLMs for 50 Languages via a Mixture of Language Family Experts</a></div>
    <div class="paper-meta">
      📅 2025-02-10
    </div>
    <details class="paper-abstract">
      Adapting medical Large Language Models to local languages can reduce barriers to accessing healthcare services, but data scarcity remains a significant challenge, particularly for low-resource languages. To address this, we first construct a high-quality medical dataset and conduct analysis to ensure its quality. In order to leverage the generalization capability of multilingual LLMs to efficiently scale to more resource-constrained languages, we explore the internal information flow of LLMs from a multilingual perspective using Mixture of Experts (MoE) modularity. Technically, we propose a novel MoE routing method that employs language-specific experts and cross-lingual routing. Inspired by circuit theory, our routing analysis revealed a Spread Out in the End information flow mechanism: while earlier layers concentrate cross-lingual information flow, the later layers exhibit language-specific divergence. This insight directly led to the development of the Post-MoE architecture, which applies sparse routing only in the later layers while maintaining dense others. Experimental results demonstrate that this approach enhances the generalization of multilingual models to other languages while preserving interpretability. Finally, to efficiently scale the model to 50 languages, we introduce the concept of language family experts, drawing on linguistic priors, which enables scaling the number of languages without adding additional parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06621v2">LinkQ: An LLM-Assisted Visual Interface for Knowledge Graph Question-Answering</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 Open-source code: https://github.com/mit-ll/linkq
    </div>
    <details class="paper-abstract">
      We present LinkQ, a system that leverages a large language model (LLM) to facilitate knowledge graph (KG) query construction through natural language question-answering. Traditional approaches often require detailed knowledge of a graph querying language, limiting the ability for users -- even experts -- to acquire valuable insights from KGs. LinkQ simplifies this process by implementing a multistep protocol in which the LLM interprets a user's question, then systematically converts it into a well-formed query. LinkQ helps users iteratively refine any open-ended questions into precise ones, supporting both targeted and exploratory analysis. Further, LinkQ guards against the LLM hallucinating outputs by ensuring users' questions are only ever answered from ground truth KG data. We demonstrate the efficacy of LinkQ through a qualitative study with five KG practitioners. Our results indicate that practitioners find LinkQ effective for KG question-answering, and desire future LLM-assisted exploratory data analysis systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00761v4">Tamper-Resistant Safeguards for Open-Weight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 Website: https://www.tamper-resistant-safeguards.com
    </div>
    <details class="paper-abstract">
      Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after hundreds of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that progress on tamper-resistance is possible, opening up a promising new avenue to improve the safety and security of open-weight LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06742v1">Gradient Multi-Normalization for Stateless and Scalable LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-10
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) typically relies on adaptive optimizers like Adam (Kingma & Ba, 2015) which store additional state information to accelerate convergence but incur significant memory overhead. Recent efforts, such as SWAN (Ma et al., 2024) address this by eliminating the need for optimizer states while achieving performance comparable to Adam via a multi-step preprocessing procedure applied to instantaneous gradients. Motivated by the success of SWAN, we introduce a novel framework for designing stateless optimizers that normalizes stochastic gradients according to multiple norms. To achieve this, we propose a simple alternating scheme to enforce the normalization of gradients w.r.t these norms. We show that our procedure can produce, up to an arbitrary precision, a fixed-point of the problem, and that SWAN is a particular instance of our approach with carefully chosen norms, providing a deeper understanding of its design. However, SWAN's computationally expensive whitening/orthogonalization step limit its practicality for large LMs. Using our principled perspective, we develop of a more efficient, scalable, and practical stateless optimizer. Our algorithm relaxes the properties of SWAN, significantly reducing its computational cost while retaining its memory efficiency, making it applicable to training large-scale models. Experiments on pre-training LLaMA models with up to 1 billion parameters demonstrate a 3X speedup over Adam with significantly reduced memory requirements, outperforming other memory-efficient baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06738v1">Resurrecting saturated LLM benchmarks with adversarial encoding</a></div>
    <div class="paper-meta">
      📅 2025-02-10
    </div>
    <details class="paper-abstract">
      Recent work showed that small changes in benchmark questions can reduce LLMs' reasoning and recall. We explore two such changes: pairing questions and adding more answer options, on three benchmarks: WMDP-bio, GPQA, and MMLU variants. We find that for more capable models, these predictably reduce performance, essentially heightening the performance ceiling of a benchmark and unsaturating it again. We suggest this approach can resurrect old benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06703v1">Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-02-10
    </div>
    <details class="paper-abstract">
      Test-Time Scaling (TTS) is an important method for improving the performance of Large Language Models (LLMs) by using additional computation during the inference phase. However, current studies do not systematically analyze how policy models, Process Reward Models (PRMs), and problem difficulty influence TTS. This lack of analysis limits the understanding and practical use of TTS methods. In this paper, we focus on two core questions: (1) What is the optimal approach to scale test-time computation across different policy models, PRMs, and problem difficulty levels? (2) To what extent can extended computation improve the performance of LLMs on complex tasks, and can smaller language models outperform larger ones through this approach? Through comprehensive experiments on MATH-500 and challenging AIME24 tasks, we have the following observations: (1) The compute-optimal TTS strategy is highly dependent on the choice of policy model, PRM, and problem difficulty. (2) With our compute-optimal TTS strategy, extremely small policy models can outperform larger models. For example, a 1B LLM can exceed a 405B LLM on MATH-500. Moreover, on both MATH-500 and AIME24, a 0.5B LLM outperforms GPT-4o, a 3B LLM surpasses a 405B LLM, and a 7B LLM beats o1 and DeepSeek-R1, while with higher inference efficiency. These findings show the significance of adapting TTS strategies to the specific characteristics of each task and model and indicate that TTS is a promising approach for enhancing the reasoning abilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18727v2">Exploring Audio Editing Features as User-Centric Privacy Defenses Against Large Language Model(LLM) Based Emotion Inference Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 Accepted for presentation(Poster) at PPAI-25: The 6th AAAI Workshop on Privacy-Preserving Artificial Intelligence
    </div>
    <details class="paper-abstract">
      The rapid proliferation of speech-enabled technologies, including virtual assistants, video conferencing platforms, and wearable devices, has raised significant privacy concerns, particularly regarding the inference of sensitive emotional information from audio data. Existing privacy-preserving methods often compromise usability and security, limiting their adoption in practical scenarios. This paper introduces a novel, user-centric approach that leverages familiar audio editing techniques, specifically pitch and tempo manipulation, to protect emotional privacy without sacrificing usability. By analyzing popular audio editing applications on Android and iOS platforms, we identified these features as both widely available and usable. We rigorously evaluated their effectiveness against a threat model, considering adversarial attacks from diverse sources, including Deep Neural Networks (DNNs), Large Language Models (LLMs), and and reversibility testing. Our experiments, conducted on three distinct datasets, demonstrate that pitch and tempo manipulation effectively obfuscates emotional data. Additionally, we explore the design principles for lightweight, on-device implementation to ensure broad applicability across various devices and platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16218v3">CoverUp: Coverage-Guided LLM-Based Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Testing is an essential part of software development. Test generation tools attempt to automate the otherwise labor-intensive task of test creation, but generating high-coverage tests remains challenging. This paper proposes CoverUp, a novel approach to driving the generation of high-coverage Python regression tests. CoverUp combines coverage analysis, code context, and feedback in prompts that iteratively guide the LLM to generate tests that improve line and branch coverage. We evaluate our prototype CoverUp implementation across a benchmark of challenging code derived from open-source Python projects and show that CoverUp substantially improves on the state of the art. Compared to CodaMosa, a hybrid search/LLM-based test generator, CoverUp achieves a per-module median line+branch coverage of 80% (vs. 47%). Compared to MuTAP, a mutation- and LLM-based test generator, CoverUp achieves an overall line+branch coverage of 90% (vs. 77%). We also demonstrate that CoverUp's performance stems not only from the LLM used but from the combined effectiveness of its components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06666v1">Automatic Evaluation of Healthcare LLMs Beyond Question-Answering</a></div>
    <div class="paper-meta">
      📅 2025-02-10
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) benchmarks are often based on open-ended or close-ended QA evaluations, avoiding the requirement of human labor. Close-ended measurements evaluate the factuality of responses but lack expressiveness. Open-ended capture the model's capacity to produce discourse responses but are harder to assess for correctness. These two approaches are commonly used, either independently or together, though their relationship remains poorly understood. This work is focused on the healthcare domain, where both factuality and discourse matter greatly. It introduces a comprehensive, multi-axis suite for healthcare LLM evaluation, exploring correlations between open and close benchmarks and metrics. Findings include blind spots and overlaps in current methodologies. As an updated sanity check, we release a new medical benchmark--CareQA--, with both open and closed variants. Finally, we propose a novel metric for open-ended evaluations --Relaxed Perplexity-- to mitigate the identified limitations.
    </details>
</div>
