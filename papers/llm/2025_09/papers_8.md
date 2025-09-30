# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14834v1">LLM Agents at the Roundtable: A Multi-Perspective and Dialectical Reasoning Framework for Essay Scoring</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has brought a new paradigm to automated essay scoring (AES), a long-standing and practical application of natural language processing in education. However, achieving human-level multi-perspective understanding and judgment remains a challenge. In this work, we propose Roundtable Essay Scoring (RES), a multi-agent evaluation framework designed to perform precise and human-aligned scoring under a zero-shot setting. RES constructs evaluator agents based on LLMs, each tailored to a specific prompt and topic context. Each agent independently generates a trait-based rubric and conducts a multi-perspective evaluation. Then, by simulating a roundtable-style discussion, RES consolidates individual evaluations through a dialectical reasoning process to produce a final holistic score that more closely aligns with human evaluation. By enabling collaboration and consensus among agents with diverse evaluation perspectives, RES outperforms prior zero-shot AES approaches. Experiments on the ASAP dataset using ChatGPT and Claude show that RES achieves up to a 34.86% improvement in average QWK over straightforward prompting (Vanilla) methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14824v1">Confirmation Bias as a Cognitive Resource in LLM-Supported Deliberation</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in group decision-making, but their influence risks fostering conformity and reducing epistemic vigilance. Drawing on the Argumentative Theory of Reasoning, we argue that confirmation bias, often seen as detrimental, can be harnessed as a resource when paired with critical evaluation. We propose a three-step process in which individuals first generate ideas independently, then use LLMs to refine and articulate them, and finally engage with LLMs as epistemic provocateurs to anticipate group critique. This framing positions LLMs as tools for scaffolding disagreement, helping individuals prepare for more productive group discussions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14803v1">OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      In online learning environments, students often lack personalized peer interactions, which play a crucial role in supporting cognitive development and learning engagement. Although previous studies have utilized large language models (LLMs) to simulate interactive dynamic learning environments for students, these interactions remain limited to conversational exchanges, lacking insights and adaptations to the learners' individualized learning and cognitive states. As a result, students' interest in discussions with AI learning companions is low, and they struggle to gain inspiration from such interactions. To address this challenge, we propose OnlineMate, a multi-agent learning companion system driven by LLMs that integrates the Theory of Mind (ToM). OnlineMate is capable of simulating peer-like agent roles, adapting to learners' cognitive states during collaborative discussions, and inferring their psychological states, such as misunderstandings, confusion, or motivation. By incorporating Theory of Mind capabilities, the system can dynamically adjust its interaction strategies to support the development of higher-order thinking and cognition. Experimental results in simulated learning scenarios demonstrate that OnlineMate effectively fosters deep learning and discussions while enhancing cognitive engagement in online educational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12536v2">jXBW: Fast Substructure Search for Large-Scale JSONL Datasets with LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      JSON Lines (JSONL) is widely used for managing large collections of semi-structured data, ranging from large language model (LLM) prompts to chemical compound records and geospatial datasets. A key operation is substructure search, which identifies all JSON objects containing a query pattern. This task underpins applications such as drug discovery (querying compounds for functional groups), prompt engineering (extracting prompts with schema fragments), and geospatial analytics (finding entities with nested attributes). However, existing methods are inefficient: traversal requires exhaustive tree matching, succinct JSON representations save space but do not accelerate search, and XML-based approaches incur conversion overhead and semantic mismatches. We present jXBW, a compressed index for efficient substructure search over JSONL. jXBW introduces three innovations: (i) a merged tree representation that consolidates repeated structures, (ii) a succinct tree index based on the eXtended Burrows--Wheeler Transform (XBW), and (iii) a three-phase algorithm for substructure search. These enable query-dependent complexity, where cost depends on query characteristics rather than dataset size, while retaining succinct space. This resolves a key bottleneck in retrieval-augmented generation (RAG) systems requiring structure-aware retrieval. Experiments on seven real datasets, including PubChem (1M compounds) and OSM geospatial data (6.6M objects), achieve up to 4,700$\times$ speedup over tree-based methods and over $6\times 10^6$ speedup relative to XML-based approaches. jXBW makes JSONL substructure search practical for the first time, opening opportunities for large-scale LLM-based analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19800v2">MOLE: Metadata Extraction and Validation in Scientific Papers Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Metadata extraction is essential for cataloging and preserving datasets, enabling effective research discovery and reproducibility, especially given the current exponential growth in scientific research. While Masader (Alyafeai et al.,2021) laid the groundwork for extracting a wide range of metadata attributes from Arabic NLP datasets' scholarly articles, it relies heavily on manual annotation. In this paper, we present MOLE, a framework that leverages Large Language Models (LLMs) to automatically extract metadata attributes from scientific papers covering datasets of languages other than Arabic. Our schema-driven methodology processes entire documents across multiple input formats and incorporates robust validation mechanisms for consistent output. Additionally, we introduce a new benchmark to evaluate the research progress on this task. Through systematic analysis of context length, few-shot learning, and web browsing integration, we demonstrate that modern LLMs show promising results in automating this task, highlighting the need for further future work improvements to ensure consistent and reliable performance. We release the code: https://github.com/IVUL-KAUST/MOLE and dataset: https://huggingface.co/datasets/IVUL-KAUST/MOLE for the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14781v1">LEAP: LLM Inference on Scalable PIM-NoC Architecture with Balanced Dataflow and Fine-Grained Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted to the 2025 International Conference on Computer-Aided Design (ICCAD'25)
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference has been a prevalent demand in daily life and industries. The large tensor sizes and computing complexities in LLMs have brought challenges to memory, computing, and databus. This paper proposes a computation/memory/communication co-designed non-von Neumann accelerator by aggregating processing-in-memory (PIM) and computational network-on-chip (NoC), termed LEAP. The matrix multiplications in LLMs are assigned to PIM or NoC based on the data dynamicity to maximize data locality. Model partition and mapping are optimized by heuristic design space exploration. Dedicated fine-grained parallelism and tiling techniques enable high-throughput dataflow across the distributed resources in PIM and NoC. The architecture is evaluated on Llama 1B/8B/13B models and shows $\sim$2.55$\times$ throughput (tokens/sec) improvement and $\sim$71.94$\times$ energy efficiency (tokens/Joule) boost compared to the A100 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01513v3">Evaluation and Facilitation of Online Discussions in the LLM Era: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 To appear in EMNLP 2025
    </div>
    <details class="paper-abstract">
      We present a survey of methods for assessing and enhancing the quality of online discussions, focusing on the potential of LLMs. While online discourses aim, at least in theory, to foster mutual understanding, they often devolve into harmful exchanges, such as hate speech, threatening social cohesion and democratic values. Recent advancements in LLMs enable artificial facilitation agents to not only moderate content, but also actively improve the quality of interactions. Our survey synthesizes ideas from NLP and Social Sciences to provide (a) a new taxonomy on discussion quality evaluation, (b) an overview of intervention and facilitation strategies, (c) along with a new taxonomy of conversation facilitation datasets, (d) an LLM-oriented roadmap of good practices and future research directions, from technological and societal perspectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14704v1">The NazoNazo Benchmark: A Cost-Effective and Extensible Test of Insight-Based Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Benchmark saturation and contamination undermine confidence in LLM evaluation. We present Nazonazo, a cost-effective and extensible benchmark built from Japanese children's riddles to test insight-based reasoning. Items are short (mostly one sentence), require no specialized domain knowledge, and can be generated at scale, enabling rapid refresh of blind sets when leakage is suspected. We evaluate 38 frontier models and 126 adults on 120 riddles. No model except for GPT-5 is comparable to human performance, which achieves a 52.9% mean accuracy. Model comparison on extended 201 items shows that reasoning models significantly outperform non-reasoning peers, while model size shows no reliable association with accuracy. Beyond aggregate accuracy, an informal candidate-tracking analysis of thought logs reveals many cases of verification failure: models often produce the correct solution among intermediate candidates yet fail to select it as the final answer, which we illustrate with representative examples observed in multiple models. Nazonazo thus offers a cost-effective, scalable, and easily renewable benchmark format that addresses the current evaluation crisis while also suggesting a recurrent meta-cognitive weakness, providing clear targets for future control and calibration methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14680v1">LEED: A Highly Efficient and Scalable LLM-Empowered Expert Demonstrations Framework for Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 5 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Multi-agent reinforcement learning (MARL) holds substantial promise for intelligent decision-making in complex environments. However, it suffers from a coordination and scalability bottleneck as the number of agents increases. To address these issues, we propose the LLM-empowered expert demonstrations framework for multi-agent reinforcement learning (LEED). LEED consists of two components: a demonstration generation (DG) module and a policy optimization (PO) module. Specifically, the DG module leverages large language models to generate instructions for interacting with the environment, thereby producing high-quality demonstrations. The PO module adopts a decentralized training paradigm, where each agent utilizes the generated demonstrations to construct an expert policy loss, which is then integrated with its own policy loss. This enables each agent to effectively personalize and optimize its local policy based on both expert knowledge and individual experience. Experimental results show that LEED achieves superior sample efficiency, time efficiency, and robust scalability compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01702v2">mdok of KInIT: Robustly Fine-tuned LLM for Binary and Multiclass AI-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 1st rank in both subtasks of the Voight-Kampff Generative AI Detection 2025 shared task (PAN@CLEF 2025)
    </div>
    <details class="paper-abstract">
      The large language models (LLMs) are able to generate high-quality texts in multiple languages. Such texts are often not recognizable by humans as generated, and therefore present a potential of LLMs for misuse (e.g., plagiarism, spams, disinformation spreading). An automated detection is able to assist humans to indicate the machine-generated texts; however, its robustness to out-of-distribution data is still challenging. This notebook describes our mdok approach in robust detection, based on fine-tuning smaller LLMs for text classification. It is applied to both subtasks of Voight-Kampff Generative AI Detection 2025, providing remarkable performance (1st rank) in both, the binary detection as well as the multiclass classification of various cases of human-AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14668v1">DeepAssert: An LLM-Aided Verification Framework with Fine-Grained Assertion Generation for Modules with Extracted Module Specifications</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Assertion-Based Verification (ABV) is a crucial method for ensuring that logic designs conform to their architectural specifications. However, existing assertion generation methods primarily rely on information either from the design specification, or register-transfer level (RTL) code. The former methods are typically limited to generating assertions for the top-level design. As the top-level design is composed of different modules without module-level specifications, they are unable to generate deep assertions that target the internal functionality of modules. The latter methods often rely on a golden RTL model, which is difficult to obtain. To address the above limitations, this paper presents a novel large language model (LLM)-aided verification framework named DeepAssert. DeepAssert is capable of analyzing the invocation relationships between modules and extracting independent specifications for each module with its I/O port information. These extracted specifications are subsequently used to guide LLMs to automatically generate fine-grained deep assertions for these modules. Our evaluation demonstrates that DeepAssert significantly outperforms existing methods such as AssertLLM and Spec2Assertion in generating high-quality deep assertions for modules. Furthermore, when integrated with these methods, DeepAssert can enhance the overall quality of the assertions generated. This allows for a more comprehensive and effective verification process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14646v1">SALT4Decompile: Inferring Source-level Abstract Logic Tree for LLM-Based Binary Decompilation</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 13 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Decompilation is widely used in reverse engineering to recover high-level language code from binary executables. While recent approaches leveraging Large Language Models (LLMs) have shown promising progress, they typically treat assembly code as a linear sequence of instructions, overlooking arbitrary jump patterns and isolated data segments inherent to binary files. This limitation significantly hinders their ability to correctly infer source code semantics from assembly code. To address this limitation, we propose \saltm, a novel binary decompilation method that abstracts stable logical features shared between binary and source code. The core idea of \saltm is to abstract selected binary-level operations, such as specific jumps, into a high-level logic framework that better guides LLMs in semantic recovery. Given a binary function, \saltm constructs a Source-level Abstract Logic Tree (\salt) from assembly code to approximate the logic structure of high-level language. It then fine-tunes an LLM using the reconstructed \salt to generate decompiled code. Finally, the output is refined through error correction and symbol recovery to improve readability and correctness. We compare \saltm to three categories of baselines (general-purpose LLMs, commercial decompilers, and decompilation methods) using three well-known datasets (Decompile-Eval, MBPP, Exebench). Our experimental results demonstrate that \saltm is highly effective in recovering the logic of the source code, significantly outperforming state-of-the-art methods (e.g., 70.4\% TCP rate on Decompile-Eval with a 10.6\% improvement). The results further validate its robustness against four commonly used obfuscation techniques. Additionally, analyses of real-world software and a user study confirm that our decompiled output offers superior assistance to human analysts in comprehending binary functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14624v1">Reveal and Release: Iterative LLM Unlearning with Self-generated Data</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language model (LLM) unlearning has demonstrated effectiveness in removing the influence of undesirable data (also known as forget data). Existing approaches typically assume full access to the forget dataset, overlooking two key challenges: (1) Forget data is often privacy-sensitive, rare, or legally regulated, making it expensive or impractical to obtain (2) The distribution of available forget data may not align with how that information is represented within the model. To address these limitations, we propose a ``Reveal-and-Release'' method to unlearn with self-generated data, where we prompt the model to reveal what it knows using optimized instructions. To fully utilize the self-generated forget data, we propose an iterative unlearning framework, where we make incremental adjustments to the model's weight space with parameter-efficient modules trained on the forget data. Experimental results demonstrate that our method balances the tradeoff between forget quality and utility preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22723v2">Zero-Shot LLMs in Human-in-the-Loop RL: Replacing Human Feedback for Reward Shaping</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 20 pages, 3 figures, 4 Tables
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) often struggles with reward misalignment, where agents optimize given rewards but fail to exhibit the desired behaviors. This arises when the reward function incentivizes proxy behaviors misaligned with the true objective. While human-in-the-loop (HITL) methods can mitigate this issue, they also introduce biases, leading to inconsistent and subjective feedback that complicates learning. To address these challenges, we propose two key contributions. First, we extend the use of zero-shot, off-the-shelf large language models (LLMs) for reward shaping beyond natural language processing (NLP) to continuous control tasks. Using LLMs as direct feedback providers eliminates the need for surrogate models trained on human feedback, which often inherit biases from training data. Second, we introduce a hybrid framework (LLM-HFBF) that enables LLMs to identify and correct biases in human feedback while incorporating this feedback into the reward shaping process. The LLM-HFBF framework creates a more balanced and reliable system by addressing both the limitations of LLMs (e.g., lack of domain-specific knowledge) and human supervision (e.g., inherent biases). By enabling human feedback bias flagging and correction, our approach improves reinforcement learning performance and reduces reliance on potentially biased human feedback. Empirical experiments show that biased human feedback significantly reduces performance, with Average Episodic Reward dropping by nearly 94% compared to unbiased approaches. In contrast, LLM-based methods sustain performance at a similar level to unbiased feedback, even in challenging edge-case scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21589v2">Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted by EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our Middo consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are coming soon. Our datasets, models, and code are publicly available at https://github.com/Word2VecT/Middo
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05282v2">ASCoT: An Adaptive Self-Correction Chain-of-Thought Method for Late-Stage Fragility in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has significantly advanced the reasoning capabilities of Large Language Models (LLMs), yet the reliability of these reasoning chains remains a critical challenge. A widely held "cascading failure" hypothesis suggests that errors are most detrimental when they occur early in the reasoning process. This paper challenges that assumption through systematic error-injection experiments, revealing a counter-intuitive phenomenon we term "Late-Stage Fragility": errors introduced in the later stages of a CoT chain are significantly more likely to corrupt the final answer than identical errors made at the beginning. To address this specific vulnerability, we introduce the Adaptive Self-Correction Chain-of-Thought (ASCoT) method. ASCoT employs a modular pipeline in which an Adaptive Verification Manager (AVM) operates first, followed by the Multi-Perspective Self-Correction Engine (MSCE). The AVM leverages a Positional Impact Score function I(k) that assigns different weights based on the position within the reasoning chains, addressing the Late-Stage Fragility issue by identifying and prioritizing high-risk, late-stage steps. Once these critical steps are identified, the MSCE applies robust, dual-path correction specifically to the failure parts. Extensive experiments on benchmarks such as GSM8K and MATH demonstrate that ASCoT achieves outstanding accuracy, outperforming strong baselines, including standard CoT. Our work underscores the importance of diagnosing specific failure modes in LLM reasoning and advocates for a shift from uniform verification strategies to adaptive, vulnerability-aware correction mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16072v2">InMind: Evaluating LLMs in Capturing and Applying Individual Human Reasoning Styles</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 EMNLP 2025 MainConference
    </div>
    <details class="paper-abstract">
      LLMs have shown strong performance on human-centric reasoning tasks. While previous evaluations have explored whether LLMs can infer intentions or detect deception, they often overlook the individualized reasoning styles that influence how people interpret and act in social contexts. Social deduction games (SDGs) provide a natural testbed for evaluating individualized reasoning styles, where different players may adopt diverse but contextually valid reasoning strategies under identical conditions. To address this, we introduce InMind, a cognitively grounded evaluation framework designed to assess whether LLMs can capture and apply personalized reasoning styles in SDGs. InMind enhances structured gameplay data with round-level strategy traces and post-game reflections, collected under both Observer and Participant modes. It supports four cognitively motivated tasks that jointly evaluate both static alignment and dynamic adaptation. As a case study, we apply InMind to the game Avalon, evaluating 11 state-of-the-art LLMs. General-purpose LLMs, even GPT-4o frequently rely on lexical cues, struggling to anchor reflections in temporal gameplay or adapt to evolving strategies. In contrast, reasoning-enhanced LLMs like DeepSeek-R1 exhibit early signs of style-sensitive reasoning. These findings reveal key limitations in current LLMs' capacity for individualized, adaptive reasoning, and position InMind as a step toward cognitively aligned human-AI interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19721v3">Unsupervised Concept Vector Extraction for Bias Control in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to perpetuate stereotypes and exhibit biases. Various strategies have been proposed to mitigate these biases, but most work studies biases as a black-box problem without considering how concepts are represented within the model. We adapt techniques from representation engineering to study how the concept of "gender" is represented within LLMs. We introduce a new method that extracts concept representations via probability weighting without labeled data and efficiently selects a steering vector for measuring and manipulating the model's representation. We develop a projection-based method that enables precise steering of model predictions and demonstrate its effectiveness in mitigating gender bias in LLMs and show that it also generalizes to racial bias. Our code is available at: https://github.com/hannahxchen/gender-bias-steering
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13869v1">Do LLMs Align Human Values Regarding Social Biases? Judging and Explaining Social Biases with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 38 pages, 31 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can lead to undesired consequences when misaligned with human values, especially in scenarios involving complex and sensitive social biases. Previous studies have revealed the misalignment of LLMs with human values using expert-designed or agent-based emulated bias scenarios. However, it remains unclear whether the alignment of LLMs with human values differs across different types of scenarios (e.g., scenarios containing negative vs. non-negative questions). In this study, we investigate the alignment of LLMs with human values regarding social biases (HVSB) in different types of bias scenarios. Through extensive analysis of 12 LLMs from four model families and four datasets, we demonstrate that LLMs with large model parameter scales do not necessarily have lower misalignment rate and attack success rate. Moreover, LLMs show a certain degree of alignment preference for specific types of scenarios and the LLMs from the same model family tend to have higher judgment consistency. In addition, we study the understanding capacity of LLMs with their explanations of HVSB. We find no significant differences in the understanding of HVSB across LLMs. We also find LLMs prefer their own generated explanations. Additionally, we endow smaller language models (LMs) with the ability to explain HVSB. The generation results show that the explanations generated by the fine-tuned smaller LMs are more readable, but have a relatively lower model agreeability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13868v1">Are Prompts All You Need? Evaluating Prompt-Based Large Language Models (LLM)s for Software Requirements Classification</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 33 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Requirements classification assigns natural language requirements to predefined classes, such as functional and non functional. Accurate classification reduces risk and improves software quality. Most existing models rely on supervised learning, which needs large labeled data that are costly, slow to create, and domain dependent; they also generalize poorly and often require retraining for each task. This study tests whether prompt based large language models can reduce data needs. We benchmark several models and prompting styles (zero shot, few shot, persona, and chain of thought) across multiple tasks on two English datasets, PROMISE and SecReq. For each task we compare model prompt configurations and then compare the best LLM setups with a strong fine tuned transformer baseline. Results show that prompt based LLMs, especially with few shot prompts, can match or exceed the baseline. Adding a persona, or persona plus chain of thought, can yield further gains. We conclude that prompt based LLMs are a practical and scalable option that reduces dependence on large annotations and can improve generalizability across tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11176v4">LogiDynamics: Unraveling the Dynamics of Inductive, Abductive and Deductive Logical Inferences in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) employ diverse logical inference mechanisms for reasoning, making the strategic optimization of these approaches critical for advancing their capabilities. This paper systematically investigate the comparative dynamics of inductive (System 1) versus abductive/deductive (System 2) inference in LLMs. We utilize a controlled analogical reasoning environment, varying modality (textual, visual, symbolic), difficulty, and task format (MCQ / free-text). Our analysis reveals System 2 pipelines generally excel, particularly in visual/symbolic modalities and harder tasks, while System 1 is competitive for textual and easier problems. Crucially, task format significantly influences their relative advantage, with System 1 sometimes outperforming System 2 in free-text rule-execution. These core findings generalize to broader in-context learning. Furthermore, we demonstrate that advanced System 2 strategies like hypothesis selection and iterative refinement can substantially scale LLM reasoning. This study offers foundational insights and actionable guidelines for strategically deploying logical inference to enhance LLM reasoning. Resources are available at https://github.com/HKUST-KnowComp/LogiDynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13790v1">Teaching According to Talents! Instruction Tuning LLMs with Competence-Aware Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Efficient instruction tuning aims to enhance the ultimate performance of large language models (LLMs) trained on a given instruction dataset. Curriculum learning as a typical data organization strategy has shown preliminary effectiveness in instruction tuning. However, current curriculum tuning methods suffer from the curriculum rigidity, since they rely solely on static heuristic difficulty metrics. These methods fail to adapt to the evolving capabilities of models during training, resulting in a fixed and potentially sub-optimal learning trajectory. To address the issue, Competence-Aware Multi-Perspective cUrriculum inStruction tuning framework termed CAMPUS is proposed. CAMPUS offers several advantages: (1) Dynamic selection for sub-curriculum. (2) Competency-aware adjustment to the curriculum schedule. (3) Multiple difficulty-based scheduling. Extensive experiments prove the superior performance of CAMPUS, compared to other state-of-the-art baselines for efficient instruction tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13765v1">TENET: An Efficient Sparsity-Aware LUT-Centric Architecture for Ternary LLM Inference On Edge</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Ternary quantization has emerged as a powerful technique for reducing both computational and memory footprint of large language models (LLM), enabling efficient real-time inference deployment without significantly compromising model accuracy. Conventional LLM inference platforms (e.g GPUs) cannot capitalize on its benefits, as they (i) lack native support for ternary arithmetic and memory specialization and (ii) remain severely under-utilized in low-batch, real-time scenarios. In this work, we propose TENET, a sparse-aware LUT-centric architecture that co-optimizes algorithm, compute, and memory for ternary LLM inference. To maximize the efficiency of Ternary Linear layer, TENET introduces a Sparse Ternary LUT (STL) core that optimizes ternary mixed-precision GEMM using a symmetric precompute lookup table. It also features Dynamic Activation N:M Sparsity to exploit the sparsity within the activation of each token. Additionally, we propose a LUT-based 64B:80B ternary weight decompression module to fully exploit the memory efficiency of ternary values. At the system level, we design a heterogeneous TENET accelerator with full programmability that integrates STL cores with high-precision cores. An associated Linear-Projection-aware Sparse Attention dataflow is introduced to optimize memory access and hardware utilization. We implement TENET accelerator prototype on both FPGA and ASIC platforms. Experiments across various model sizes and workloads demonstrate that TENET-FPGA and TENET-ASIC improve energy efficiency by 4.3$\times$ and 21.1$\times$, respectively, compared to the A100 GPU. Furthermore, TENET-ASIC achieves a 2.7$\times$ average speedup compared to the A100 GPU in end-to-end inference latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15495v3">SynthCoder: A Synthetical Strategy to Tune LLMs for Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Code completion is a prominent application of Large Language Models (LLMs) in software engineering. Due to the near real-time response requirements of this task, base models with small to medium-sized parameters are typically employed, supplemented by various optimization and post-training techniques. However, these optimization methods often have trade-offs, leading to a seesaw effect where performance improvements on certain datasets or metrics are accompanied by degradations on others -- sometimes even falling below the baseline model's performance. This paper proposes SynthCoder, a model that integrates leading industry practices to achieve state-of-the-art performance on the Fill-in-the-Middle (FIM) code completion task. In specific, we first construct a diverse dataset by combining Abstract Syntax Tree (AST) node extraction with heuristics that simulate developer behavior. Then we enrich our training corpus with cross-file contextual information using the BM25 algorithm and call graphs, enhancing the model's ability to perform code completion in both file-level and repository-level scenarios. As the last step, we employ a two-stage training process using the Seed-Coder-8B-Base as the base model. First, we fine-tune the model using Curriculum Learning technology. Following this, we perform alignment using Direct Preference Optimization (DPO) with preference pairs generated through Rejection Sampling. Experimental results demonstrate that our final model excels on mainstream repository-level code completion benchmarks, including aiXcoder, ExecRepoBench, CrossCodeEval, and CoLT. Furthermore, our carefully curated training set effectively mitigates the model's tendency to just repeat existing code, a common issue existing in various code completion models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19894v4">Semantic Alignment-Enhanced Code Translation via an LLM-Based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Code translation converts code from one programming language to another while maintaining its original functionality, which is crucial for software migration, system refactoring, and cross-platform development. Traditional rule-based methods rely on manually-written rules, which can be time-consuming and often result in less readable code. To overcome this, learning-based methods have been developed, leveraging parallel data to train models for automated code translation. More recently, the advance of Large Language Models (LLMs) further boosts learning-based code translation. Although promising, LLM-translated program still suffers from diverse quality issues (e.g., syntax errors and semantic errors). In particular, it can be challenging for LLMs to self-debug these errors when simply provided with the corresponding error messages. In this work, we propose a novel LLM-based multi-agent system TRANSAGENT, which enhances LLM-based code translation by fixing the syntax errors and semantic errors with the synergy between four LLM-based agents, including Initial Code Translator, Syntax Error Fixer, Code Aligner, and Semantic Error Fixer. The main insight of TRANSAGENT is to first localize the error code block in the target program based on the execution alignment between the target and source program, which can narrow down the fixing space and thus lower down the fixing difficulties. To evaluate TRANSAGENT, we first construct a new benchmark from recent programming tasks to mitigate the potential data leakage issue. On our benchmark, TRANSAGENT outperforms the latest LLM-based code translation technique UniTrans in both translation effectiveness and efficiency; additionally, our evaluation on different LLMs show the generalization of TRANSAGENT and our ablation study shows the contribution of each agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13710v1">CompAir: Synergizing Complementary PIMs and In-Transit NoC Computation for Efficient LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has revolutionized various aspects of human life, yet their immense computational and energy demands pose significant challenges for efficient inference. The memory wall, the growing processor-memory speed disparity, remains a critical bottleneck for LLM. Process-In-Memory (PIM) architectures overcome limitations by co-locating compute units with memory, leveraging 5-20$\times$ higher internal bandwidth and enabling greater energy efficiency than GPUs. However, existing PIMs struggle to balance flexibility, performance, and cost-efficiency for LLMs' dynamic memory-compute patterns and operator diversity. DRAM-PIM suffers from inter-bank communication overhead despite its vector parallelism. SRAM-PIM offers sub-10ns latency for matrix operation but is constrained by limited capacity. This work introduces CompAir, a novel PIM architecture that integrates DRAM-PIM and SRAM-PIM with hybrid bonding, enabling efficient linear computations while unlocking multi-granularity data pathways. We further develop CompAir-NoC, an advanced network-on-chip with an embedded arithmetic logic unit that performs non-linear operations during data movement, simultaneously reducing communication overhead and area cost. Finally, we develop a hierarchical Instruction Set Architecture that ensures both flexibility and programmability of the hybrid PIM. Experimental results demonstrate that CompAir achieves 1.83-7.98$\times$ prefill and 1.95-6.28$\times$ decode improvement over the current state-of-the-art fully PIM architecture. Compared to the hybrid A100 and HBM-PIM system, CompAir achieves 3.52$\times$ energy consumption reduction with comparable throughput. This work represents the first systematic exploration of hybrid DRAM-PIM and SRAM-PIM architectures with in-network computation capabilities, offering a high-efficiency solution for LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07445v2">Forget What You Know about LLMs Evaluations -- LLMs are Like a Chameleon</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often appear to excel on public benchmarks, but these high scores may mask an overreliance on dataset-specific surface cues rather than true language understanding. We introduce the Chameleon Benchmark Overfit Detector (C-BOD), a meta-evaluation framework that systematically distorts benchmark prompts via a parametric transformation and detects overfitting of LLMs. By rephrasing inputs while preserving their semantic content and labels, C-BOD exposes whether a model's performance is driven by memorized patterns. Evaluated on the MMLU benchmark using 26 leading LLMs, our method reveals an average performance degradation of 2.15% under modest perturbations, with 20 out of 26 models exhibiting statistically significant differences. Notably, models with higher baseline accuracy exhibit larger performance differences under perturbation, and larger LLMs tend to be more sensitive to rephrasings, indicating that both cases may overrely on fixed prompt patterns. In contrast, the Llama family and models with lower baseline accuracy show insignificant degradation, suggesting reduced dependency on superficial cues. Moreover, C-BOD's dataset- and model-agnostic design allows easy integration into training pipelines to promote more robust language understanding. Our findings challenge the community to look beyond leaderboard scores and prioritize resilience and generalization in LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13694v1">StreamTensor: Make Tensors Stream in Dataflow Accelerators for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Accepted by MICRO'25
    </div>
    <details class="paper-abstract">
      Efficient execution of deep learning workloads on dataflow architectures is crucial for overcoming memory bottlenecks and maximizing performance. While streaming intermediate results between computation kernels can significantly improve efficiency, existing approaches struggle with inter-kernel correlations, external memory access management, and buffer optimization. In this work, we propose StreamTensor, a compiler framework that automatically constructs and optimizes stream-based dataflow accelerators. StreamTensor introduces a novel iterative tensor type system to explicitly encode stream layouts, enabling seamless kernel fusion, buffer allocation, and memory optimization. By systematically exploring three hierarchical design spaces, including tensor tiling, kernel fusion, and resource allocation, StreamTensor balances computational intensity, memory efficiency, and data streaming to maximize performance. Based on FPGA evaluations on Large Language Models (LLM), StreamTensor achieves up to 0.76x and 0.64x lower latency compared to the state-of-the-art FPGA LLM accelerators and GPUs, and up to 1.99x higher energy efficiency compared to GPUs, making it a promising approach for scalable dataflow-based deep learning acceleration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13691v1">SPAR: Scalable LLM-based PDDL Domain Generation for Aerial Robotics</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      We investigate the problem of automatic domain generation for the Planning Domain Definition Language (PDDL) using Large Language Models (LLMs), with a particular focus on unmanned aerial vehicle (UAV) tasks. Although PDDL is a widely adopted standard in robotic planning, manually designing domains for diverse applications such as surveillance, delivery, and inspection is labor-intensive and error-prone, which hinders adoption and real-world deployment. To address these challenges, we propose SPAR, a framework that leverages the generative capabilities of LLMs to automatically produce valid, diverse, and semantically accurate PDDL domains from natural language input. To this end, we first introduce a systematically formulated and validated UAV planning dataset, consisting of ground-truth PDDL domains and associated problems, each paired with detailed domain and action descriptions. Building on this dataset, we design a prompting framework that generates high-quality PDDL domains from language input. The generated domains are evaluated through syntax validation, executability, feasibility, and interpretability. Overall, this work demonstrates that LLMs can substantially accelerate the creation of complex planning domains, providing a reproducible dataset and evaluation pipeline that enables application experts without prior experience to leverage it for practical tasks and advance future research in aerial robotics and automated planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09061v3">EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 4 figures, 7 pages, IEEE conference template
    </div>
    <details class="paper-abstract">
      This paper introduces EdgeProfiler, a fast profiling framework designed for evaluating lightweight Large Language Models (LLMs) on edge systems. While LLMs offer remarkable capabilities in natural language understanding and generation, their high computational, memory, and power requirements often confine them to cloud environments. EdgeProfiler addresses these challenges by providing a systematic methodology for assessing LLM performance in resource-constrained edge settings. The framework profiles compact LLMs, including TinyLLaMA, Gemma3.1B, Llama3.2-1B, and DeepSeek-r1-1.5B, using aggressive quantization techniques and strict memory constraints. Analytical modeling is used to estimate latency, FLOPs, and energy consumption. The profiling reveals that 4-bit quantization reduces model memory usage by approximately 60-70%, while maintaining accuracy within 2-5% of full-precision baselines. Inference speeds are observed to improve by 2-3x compared to FP16 baselines across various edge devices. Power modeling estimates a 35-50% reduction in energy consumption for INT4 configurations, enabling practical deployment on hardware such as Raspberry Pi 4/5 and Jetson Orin Nano Super. Our findings emphasize the importance of efficient profiling tailored to lightweight LLMs in edge environments, balancing accuracy, energy efficiency, and computational feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13664v1">Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 To be appeared in EMNLP 2025 (main)
    </div>
    <details class="paper-abstract">
      Ambiguity is pervasive in real-world questions, yet large language models (LLMs) often respond with confident answers rather than seeking clarification. In this work, we show that question ambiguity is linearly encoded in the internal representations of LLMs and can be both detected and controlled at the neuron level. During the model's pre-filling stage, we identify that a small number of neurons, as few as one, encode question ambiguity information. Probes trained on these Ambiguity-Encoding Neurons (AENs) achieve strong performance on ambiguity detection and generalize across datasets, outperforming prompting-based and representation-based baselines. Layerwise analysis reveals that AENs emerge from shallow layers, suggesting early encoding of ambiguity signals in the model's processing pipeline. Finally, we show that through manipulating AENs, we can control LLM's behavior from direct answering to abstention. Our findings reveal that LLMs form compact internal representations of question ambiguity, enabling interpretable and controllable behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13642v1">LLM-I: LLMs are Naturally Interleaved Multimodal Creators</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      We propose LLM-Interleaved (LLM-I), a flexible and dynamic framework that reframes interleaved image-text generation as a tool-use problem. LLM-I is designed to overcome the "one-tool" bottleneck of current unified models, which are limited to synthetic imagery and struggle with tasks requiring factual grounding or programmatic precision. Our framework empowers a central LLM or MLLM agent to intelligently orchestrate a diverse toolkit of specialized visual tools, including online image search, diffusion-based generation, code execution, and image editing. The agent is trained to select and apply these tools proficiently via a Reinforcement Learning (RL) framework that features a hybrid reward system combining rule-based logic with judgments from LLM and MLLM evaluators. Trained on a diverse new dataset using four different model backbones, LLM-I demonstrates state-of-the-art performance, outperforming existing methods by a large margin across four benchmarks. We also introduce a novel test-time scaling strategy that provides further performance gains. Project Page: https://github.com/ByteDance-BandAI/LLM-I.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13624v1">Latent Traits and Cross-Task Transfer: Deconstructing Dataset Interactions in LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Camera-ready version. Accepted to appear in the proceedings of the 14th Joint Conference on Lexical and Computational Semantics (*SEM 2025)
    </div>
    <details class="paper-abstract">
      Large language models are increasingly deployed across diverse applications. This often includes tasks LLMs have not encountered during training. This implies that enumerating and obtaining the high-quality training data for all tasks is infeasible. Thus, we often need to rely on transfer learning using datasets with different characteristics, and anticipate out-of-distribution requests. Motivated by this practical need, we propose an analysis framework, building a transfer learning matrix and dimensionality reduction, to dissect these cross-task interactions. We train and analyze 10 models to identify latent abilities (e.g., Reasoning, Sentiment Classification, NLU, Arithmetic) and discover the side effects of the transfer learning. Our findings reveal that performance improvements often defy explanations based on surface-level dataset similarity or source data quality. Instead, hidden statistical factors of the source dataset, such as class distribution and generation length proclivities, alongside specific linguistic features, are actually more influential. This work offers insights into the complex dynamics of transfer learning, paving the way for more predictable and effective LLM adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11963v3">NeedleBench: Evaluating LLM Retrieval and Reasoning Across Varying Information Densities</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 v3: Revisions with added experiments, clarifications, and related work updates
    </div>
    <details class="paper-abstract">
      The capability of large language models to handle long-context information is crucial across various real-world applications. Existing evaluation methods often rely either on real-world long texts, making it difficult to exclude the influence of models' inherent knowledge, or introduce irrelevant filler content to artificially achieve target lengths, reducing assessment effectiveness. To address these limitations, we introduce NeedleBench, a synthetic framework for assessing retrieval and reasoning performance in bilingual long-context tasks with adaptive context lengths. NeedleBench systematically embeds key data points at varying depths to rigorously test model capabilities. Tasks are categorized into two scenarios: information-sparse, featuring minimal relevant details within extensive irrelevant text to simulate simple retrieval tasks; and information-dense (the Ancestral Trace Challenge), where relevant information is continuously distributed throughout the context to simulate complex reasoning tasks. Our experiments reveal that although recent reasoning models like Deepseek-R1 and OpenAI's o3 excel in mathematical reasoning, they struggle with continuous retrieval and reasoning in information-dense scenarios, even at shorter context lengths. We also characterize a phenomenon termed 'under-thinking', where models prematurely conclude reasoning despite available information. NeedleBench thus provides critical insights and targeted tools essential for evaluating and improving LLMs' long-context capabilities. All resources are available at OpenCompass: https://github.com/open-compass/opencompass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13603v1">Modernizing Facebook Scoped Search: Keyword and Embedding Hybrid Retrieval with LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 5 Pages, work done as Yongye Su's internship project at Meta
    </div>
    <details class="paper-abstract">
      Beyond general web-scale search, social network search uniquely enables users to retrieve information and discover potential connections within their social context. We introduce a framework of modernized Facebook Group Scoped Search by blending traditional keyword-based retrieval with embedding-based retrieval (EBR) to improve the search relevance and diversity of search results. Our system integrates semantic retrieval into the existing keyword search pipeline, enabling users to discover more contextually relevant group posts. To rigorously assess the impact of this blended approach, we introduce a novel evaluation framework that leverages large language models (LLMs) to perform offline relevance assessments, providing scalable and consistent quality benchmarks. Our results demonstrate that the blended retrieval system significantly enhances user engagement and search quality, as validated by both online metrics and LLM-based evaluation. This work offers practical insights for deploying and evaluating advanced retrieval systems in large-scale, real-world social platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14483v1">An LLM-based multi-agent framework for agile effort estimation</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Submitted to ASE'25
    </div>
    <details class="paper-abstract">
      Effort estimation is a crucial activity in agile software development, where teams collaboratively review, discuss, and estimate the effort required to complete user stories in a product backlog. Current practices in agile effort estimation heavily rely on subjective assessments, leading to inaccuracies and inconsistencies in the estimates. While recent machine learning-based methods show promising accuracy, they cannot explain or justify their estimates and lack the capability to interact with human team members. Our paper fills this significant gap by leveraging the powerful capabilities of Large Language Models (LLMs). We propose a novel LLM-based multi-agent framework for agile estimation that not only can produce estimates, but also can coordinate, communicate and discuss with human developers and other agents to reach a consensus. Evaluation results on a real-life dataset show that our approach outperforms state-of-the-art techniques across all evaluation metrics in the majority of the cases. Our human study with software development practitioners also demonstrates an overwhelmingly positive experience in collaborating with our agents in agile effort estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14478v1">Estimating Semantic Alphabet Size for LLM Uncertainty Quantification</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Many black-box techniques for quantifying the uncertainty of large language models (LLMs) rely on repeated LLM sampling, which can be computationally expensive. Therefore, practical applicability demands reliable estimation from few samples. Semantic entropy (SE) is a popular sample-based uncertainty estimator with a discrete formulation attractive for the black-box setting. Recent extensions of semantic entropy exhibit improved LLM hallucination detection, but do so with less interpretable methods that admit additional hyperparameters. For this reason, we revisit the canonical discrete semantic entropy estimator, finding that it underestimates the "true" semantic entropy, as expected from theory. We propose a modified semantic alphabet size estimator, and illustrate that using it to adjust discrete semantic entropy for sample coverage results in more accurate semantic entropy estimation in our setting of interest. Furthermore, our proposed alphabet size estimator flags incorrect LLM responses as well or better than recent top-performing approaches, with the added benefit of remaining highly interpretable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14464v1">Not What the Doctor Ordered: Surveying LLM-based De-identification and Quantifying Clinical Information Loss</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      De-identification in the healthcare setting is an application of NLP where automated algorithms are used to remove personally identifying information of patients (and, sometimes, providers). With the recent rise of generative large language models (LLMs), there has been a corresponding rise in the number of papers that apply LLMs to de-identification. Although these approaches often report near-perfect results, significant challenges concerning reproducibility and utility of the research papers persist. This paper identifies three key limitations in the current literature: inconsistent reporting metrics hindering direct comparisons, the inadequacy of traditional classification metrics in capturing errors which LLMs may be more prone to (i.e., altering clinically relevant information), and lack of manual validation of automated metrics which aim to quantify these errors. To address these issues, we first present a survey of LLM-based de-identification research, highlighting the heterogeneity in reporting standards. Second, we evaluated a diverse set of models to quantify the extent of inappropriate removal of clinical information. Next, we conduct a manual validation of an existing evaluation metric to measure the removal of clinical information, employing clinical experts to assess their efficacy. We highlight poor performance and describe the inherent limitations of such metrics in identifying clinically significant changes. Lastly, we propose a novel methodology for the detection of clinically relevant information removal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12476v2">Audited Reasoning Refinement: Fine-Tuning Language Models via LLM-Guided Step-Wise Evaluation and Correction</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Training a task-specific small reasoning model is challenging when direct human supervision or high-quality labels are scarce. However, LLMs with reasoning capabilities produce abundant intermediate reasoning traces that can be systematically refined to create effective supervision signals. We propose Reason-Refine-then-Align (R2tA), which turns refined model rationales into supervision for training task-specific reasoning models. Our method generates initial reasoning and responses from an open-source base model on task-specific inputs, then refines these traces, fixing hallucinations and inconsistencies, to form a high-fidelity dataset. We perform a two-stage alignment, supervised fine-tuning (SFT), followed by direct preference optimization (DPO) to calibrate the model's intermediate reasoning with human-validated conceptual preferences and then condition the final output on that aligned reasoning. As a case study, we apply R2tA to evaluate extended entity relationship diagrams (EERDs) in database system design, a structurally complex task where prompt-only methods miss or hallucinate errors. We curated a dataset of 600 EERD variants (train/test split of 450/150, respectively) with induced mistakes spanning 11 categories. Empirical evaluation suggests R2tA provides a practical, cost-effective path to scalable LLM adaptation in data-scarce domains, enabling reproducible AI tools for education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14456v1">Correct-Detect: Balancing Performance and Ambiguity Through the Lens of Coreference Resolution in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are intended to reflect human linguistic competencies. But humans have access to a broad and embodied context, which is key in detecting and resolving linguistic ambiguities, even in isolated text spans. A foundational case of semantic ambiguity is found in the task of coreference resolution: how is a pronoun related to an earlier person mention? This capability is implicit in nearly every downstream task, and the presence of ambiguity at this level can alter performance significantly. We show that LLMs can achieve good performance with minimal prompting in both coreference disambiguation and the detection of ambiguity in coreference, however, they cannot do both at the same time. We present the CORRECT-DETECT trade-off: though models have both capabilities and deploy them implicitly, successful performance balancing these two abilities remains elusive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14448v1">VCBench: Benchmarking LLMs in Venture Capital</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Benchmarks such as SWE-bench and ARC-AGI demonstrate how shared datasets accelerate progress toward artificial general intelligence (AGI). We introduce VCBench, the first benchmark for predicting founder success in venture capital (VC), a domain where signals are sparse, outcomes are uncertain, and even top investors perform modestly. At inception, the market index achieves a precision of 1.9%. Y Combinator outperforms the index by a factor of 1.7x, while tier-1 firms are 2.9x better. VCBench provides 9,000 anonymized founder profiles, standardized to preserve predictive features while resisting identity leakage, with adversarial tests showing more than 90% reduction in re-identification risk. We evaluate nine state-of-the-art large language models (LLMs). DeepSeek-V3 delivers over six times the baseline precision, GPT-4o achieves the highest F0.5, and most models surpass human benchmarks. Designed as a public and evolving resource available at vcbench.com, VCBench establishes a community-driven standard for reproducible and privacy-preserving evaluation of AGI in early-stage venture forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19477v3">Judging with Many Minds: Do More Perspectives Mean Less Prejudice? On Bias Amplifications and Resistance in Multi-Agent Based LLM-as-Judge</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      LLM-as-Judge has emerged as a scalable alternative to human evaluation, enabling large language models (LLMs) to provide reward signals in trainings. While recent work has explored multi-agent extensions such as multi-agent debate and meta-judging to enhance evaluation quality, the question of how intrinsic biases manifest in these settings remains underexplored. In this study, we conduct a systematic analysis of four diverse bias types: position bias, verbosity bias, chain-of-thought bias, and bandwagon bias. We evaluate these biases across two widely adopted multi-agent LLM-as-Judge frameworks: Multi-Agent-Debate and LLM-as-Meta-Judge. Our results show that debate framework amplifies biases sharply after the initial debate, and this increased bias is sustained in subsequent rounds, while meta-judge approaches exhibit greater resistance. We further investigate the incorporation of PINE, a leading single-agent debiasing method, as a bias-free agent within these systems. The results reveal that this bias-free agent effectively reduces biases in debate settings but provides less benefit in meta-judge scenarios. Our work provides a comprehensive study of bias behavior in multi-agent LLM-as-Judge systems and highlights the need for targeted bias mitigation strategies in collaborative evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14405v1">Adding LLMs to the psycholinguistic norming toolbox: A practical guide to getting the most out of human ratings</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Word-level psycholinguistic norms lend empirical support to theories of language processing. However, obtaining such human-based measures is not always feasible or straightforward. One promising approach is to augment human norming datasets by using Large Language Models (LLMs) to predict these characteristics directly, a practice that is rapidly gaining popularity in psycholinguistics and cognitive science. However, the novelty of this approach (and the relative inscrutability of LLMs) necessitates the adoption of rigorous methodologies that guide researchers through this process, present the range of possible approaches, and clarify limitations that are not immediately apparent, but may, in some cases, render the use of LLMs impractical. In this work, we present a comprehensive methodology for estimating word characteristics with LLMs, enriched with practical advice and lessons learned from our own experience. Our approach covers both the direct use of base LLMs and the fine-tuning of models, an alternative that can yield substantial performance gains in certain scenarios. A major emphasis in the guide is the validation of LLM-generated data with human "gold standard" norms. We also present a software framework that implements our methodology and supports both commercial and open-weight models. We illustrate the proposed approach with a case study on estimating word familiarity in English. Using base models, we achieved a Spearman correlation of 0.8 with human ratings, which increased to 0.9 when employing fine-tuned models. This methodology, framework, and set of best practices aim to serve as a reference for future research on leveraging LLMs for psycholinguistic and lexical studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14404v1">A Taxonomy of Prompt Defects in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become key components of modern software, with prompts acting as their de-facto programming interface. However, prompt design remains largely empirical and small mistakes can cascade into unreliable, insecure, or inefficient behavior. This paper presents the first systematic survey and taxonomy of prompt defects, recurring ways that prompts fail to elicit their intended behavior from LLMs. We organize defects along six dimensions: (1) Specification and Intent, (2) Input and Content, (3) Structure and Formatting, (4) Context and Memory, (5) Performance and Efficiency, and (6) Maintainability and Engineering. Each dimension is refined into fine-grained subtypes, illustrated with concrete examples and root cause analysis. Grounded in software engineering principles, we show how these defects surface in real development workflows and examine their downstream effects. For every subtype, we distill mitigation strategies that span emerging prompt engineering patterns, automated guardrails, testing harnesses, and evaluation frameworks. We then summarize these strategies in a master taxonomy that links defect, impact, and remedy. We conclude with open research challenges and a call for rigorous engineering-oriented methodologies to ensure that LLM-driven systems are dependable by design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14391v1">Q-ROAR: Outlier-Aware Rescaling for RoPE Position Interpolation in Quantized Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Extending LLM context windows is crucial for long range tasks. RoPE-based position interpolation (PI) methods like linear and frequency-aware scaling extend input lengths without retraining, while post-training quantization (PTQ) enables practical deployment. We show that combining PI with PTQ degrades accuracy due to coupled effects long context aliasing, dynamic range dilation, axis grid anisotropy, and outlier shifting that induce position-dependent logit noise. We provide the first systematic analysis of PI plus PTQ and introduce two diagnostics: Interpolation Pressure (per-band phase scaling sensitivity) and Tail Inflation Ratios (outlier shift from short to long contexts). To address this, we propose Q-ROAR, a RoPE-aware, weight-only stabilization that groups RoPE dimensions into a few frequency bands and performs a small search over per-band scales for W_Q,W_K, with an optional symmetric variant to preserve logit scale. The diagnostics guided search uses a tiny long-context dev set and requires no fine-tuning, kernel, or architecture changes. Empirically, Q-ROAR recovers up to 0.7% accuracy on standard tasks and reduces GovReport perplexity by more than 10%, while preserving short-context performance and compatibility with existing inference stacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14335v1">Beyond Classification: Evaluating LLMs for Fine-Grained Automatic Malware Behavior Auditing</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Automated malware classification has achieved strong detection performance. Yet, malware behavior auditing seeks causal and verifiable explanations of malicious activities -- essential not only to reveal what malware does but also to substantiate such claims with evidence. This task is challenging, as adversarial intent is often hidden within complex, framework-heavy applications, making manual auditing slow and costly. Large Language Models (LLMs) could help address this gap, but their auditing potential remains largely unexplored due to three limitations: (1) scarce fine-grained annotations for fair assessment; (2) abundant benign code obscuring malicious signals; and (3) unverifiable, hallucination-prone outputs undermining attribution credibility. To close this gap, we introduce MalEval, a comprehensive framework for fine-grained Android malware auditing, designed to evaluate how effectively LLMs support auditing under real-world constraints. MalEval provides expert-verified reports and an updated sensitive API list to mitigate ground truth scarcity and reduce noise via static reachability analysis. Function-level structural representations serve as intermediate attribution units for verifiable evaluation. Building on this, we define four analyst-aligned tasks -- function prioritization, evidence attribution, behavior synthesis, and sample discrimination -- together with domain-specific metrics and a unified workload-oriented score. We evaluate seven widely used LLMs on a curated dataset of recent malware and misclassified benign apps, offering the first systematic assessment of their auditing capabilities. MalEval reveals both promising potential and critical limitations across audit stages, providing a reproducible benchmark and foundation for future research on LLM-enhanced malware behavior auditing. MalEval is publicly available at https://github.com/ZhengXR930/MalEval.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14297v1">A Simple and Efficient Jailbreak Method Exploiting LLMs' Helpfulness</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Safety alignment aims to prevent Large Language Models (LLMs) from responding to harmful queries. To strengthen safety protections, jailbreak methods are developed to simulate malicious attacks and uncover vulnerabilities. In this paper, we introduce HILL (Hiding Intention by Learning from LLMs), a novel jailbreak approach that systematically transforms imperative harmful requests into learning-style questions with only straightforward hypotheticality indicators. Further, we introduce two new metrics to thoroughly evaluate the utility of jailbreak methods. Experiments on the AdvBench dataset across a wide range of models demonstrate HILL's strong effectiveness, generalizability, and harmfulness. It achieves top attack success rates on the majority of models and across malicious categories while maintaining high efficiency with concise prompts. Results of various defense methods show the robustness of HILL, with most defenses having mediocre effects or even increasing the attack success rates. Moreover, the assessment on our constructed safe prompts reveals inherent limitations of LLMs' safety mechanisms and flaws in defense methods. This work exposes significant vulnerabilities of safety measures against learning-style elicitation, highlighting a critical challenge of balancing helpfulness and safety alignments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15234v1">Exploring the Capabilities of LLM Encoders for Image-Text Retrieval in Chest X-rays</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 24 pages, 2 figures, under review
    </div>
    <details class="paper-abstract">
      Vision-language pretraining has advanced image-text alignment, yet progress in radiology remains constrained by the heterogeneity of clinical reports, including abbreviations, impression-only notes, and stylistic variability. Unlike general-domain settings where more data often leads to better performance, naively scaling to large collections of noisy reports can plateau or even degrade model learning. We ask whether large language model (LLM) encoders can provide robust clinical representations that transfer across diverse styles and better guide image-text alignment. We introduce LLM2VEC4CXR, a domain-adapted LLM encoder for chest X-ray reports, and LLM2CLIP4CXR, a dual-tower framework that couples this encoder with a vision backbone. LLM2VEC4CXR improves clinical text understanding over BERT-based baselines, handles abbreviations and style variation, and achieves strong clinical alignment on report-level metrics. LLM2CLIP4CXR leverages these embeddings to boost retrieval accuracy and clinically oriented scores, with stronger cross-dataset generalization than prior medical CLIP variants. Trained on 1.6M CXR studies from public and private sources with heterogeneous and noisy reports, our models demonstrate that robustness -- not scale alone -- is the key to effective multimodal learning. We release models to support further research in medical image-text representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14233v1">Apertus: Democratizing Open and Compliant LLMs for Global Language Environments</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting robots.txt exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23804v2">Calibrating LLMs for Text-to-SQL Parsing by Leveraging Sub-clause Frequencies</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 EMNLP 2025 main conference
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) achieve strong performance on text-to-SQL parsing, they sometimes exhibit unexpected failures in which they are confidently incorrect. Building trustworthy text-to-SQL systems thus requires eliciting reliable uncertainty measures from the LLM. In this paper, we study the problem of providing a calibrated confidence score that conveys the likelihood of an output query being correct. Our work is the first to establish a benchmark for post-hoc calibration of LLM-based text-to-SQL parsing. In particular, we show that Platt scaling, a canonical method for calibration, provides substantial improvements over directly using raw model output probabilities as confidence scores. Furthermore, we propose a method for text-to-SQL calibration that leverages the structured nature of SQL queries to provide more granular signals of correctness, named "sub-clause frequency" (SCF) scores. Using multivariate Platt scaling (MPS), our extension of the canonical Platt scaling technique, we combine individual SCF scores into an overall accurate and calibrated score. Empirical evaluation on two popular text-to-SQL datasets shows that our approach of combining MPS and SCF yields further improvements in calibration and the related task of error detection over traditional Platt scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14180v1">Synthesizing Behaviorally-Grounded Reasoning Chains: A Data-Generation Framework for Personal Finance LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 24 pages, 11 figures. The paper presents a novel framework for generating a personal finance dataset. The resulting fine-tuned model and dataset are publicly available
    </div>
    <details class="paper-abstract">
      Personalized financial advice requires consideration of user goals, constraints, risk tolerance, and jurisdiction. Prior LLM work has focused on support systems for investors and financial planners. Simultaneously, numerous recent studies examine broader personal finance tasks, including budgeting, debt management, retirement, and estate planning, through agentic pipelines that incur high maintenance costs, yielding less than 25% of their expected financial returns. In this study, we introduce a novel and reproducible framework that integrates relevant financial context with behavioral finance studies to construct supervision data for end-to-end advisors. Using this framework, we create a 19k sample reasoning dataset and conduct a comprehensive fine-tuning of the Qwen-3-8B model on the dataset. Through a held-out test split and a blind LLM-jury study, we demonstrate that through careful data curation and behavioral integration, our 8B model achieves performance comparable to significantly larger baselines (14-32B parameters) across factual accuracy, fluency, and personalization metrics while incurring 80% lower costs than the larger counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20781v2">Using LLMs in Generating Design Rationale for Software Architecture Decisions</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 38 pages, 5 images, 9 tables, Manuscript revision submitted to a journal (2025)
    </div>
    <details class="paper-abstract">
      Design Rationale (DR) for software architecture decisions refers to the reasoning underlying architectural choices, which provides valuable insights into the different phases of the architecting process throughout software development. However, in practice, DR is often inadequately documented due to a lack of motivation and effort from developers. With the recent advancements in Large Language Models (LLMs), their capabilities in text comprehension, reasoning, and generation may enable the generation and recovery of DR for architecture decisions. In this study, we evaluated the performance of LLMs in generating DR for architecture decisions. First, we collected 50 Stack Overflow (SO) posts, 25 GitHub issues, and 25 GitHub discussions related to architecture decisions to construct a dataset of 100 architecture-related problems. Then, we selected five LLMs to generate DR for the architecture decisions with three prompting strategies, including zero-shot, chain of thought (CoT), and LLM-based agents. With the DR provided by human experts as ground truth, the Precision of LLM-generated DR with the three prompting strategies ranges from 0.267 to 0.278, Recall from 0.627 to 0.715, and F1-score from 0.351 to 0.389. Additionally, 64.45% to 69.42% of the arguments of DR not mentioned by human experts are also helpful, 4.12% to 4.87% of the arguments have uncertain correctness, and 1.59% to 3.24% of the arguments are potentially misleading. To further understand the trustworthiness and applicability of LLM-generated DR in practice, we conducted semi-structured interviews with six practitioners. Based on the experimental and interview results, we discussed the pros and cons of the three prompting strategies, the strengths and limitations of LLM-generated DR, and the implications for the practical use of LLM-generated DR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14169v1">TopoSizing: An LLM-aided Framework of Topology-based Understanding and Sizing for AMS Circuits</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Analog and mixed-signal circuit design remains challenging due to the shortage of high-quality data and the difficulty of embedding domain knowledge into automated flows. Traditional black-box optimization achieves sampling efficiency but lacks circuit understanding, which often causes evaluations to be wasted in low-value regions of the design space. In contrast, learning-based methods embed structural knowledge but are case-specific and costly to retrain. Recent attempts with large language models show potential, yet they often rely on manual intervention, limiting generality and transparency. We propose TopoSizing, an end-to-end framework that performs robust circuit understanding directly from raw netlists and translates this knowledge into optimization gains. Our approach first applies graph algorithms to organize circuits into a hierarchical device-module-stage representation. LLM agents then execute an iterative hypothesis-verification-refinement loop with built-in consistency checks, producing explicit annotations. Verified insights are integrated into Bayesian optimization through LLM-guided initial sampling and stagnation-triggered trust-region updates, improving efficiency while preserving feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18325v3">Understanding and Mitigating Overrefusal in LLMs from an Unveiling Perspective of Safety Decision Boundary</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they often refuse to answer legitimate queries--a phenomenon known as overrefusal. Overrefusal typically stems from over-conservative safety alignment, causing models to treat many reasonable prompts as potentially risky. To systematically understand this issue, we probe and leverage the models' safety decision boundaries to analyze and mitigate overrefusal. Our findings reveal that overrefusal is closely tied to misalignment at these boundary regions, where models struggle to distinguish subtle differences between benign and harmful content. Building on these insights, we present RASS, an automated framework for prompt generation and selection that strategically targets overrefusal prompts near the safety boundary. By harnessing steering vectors in the representation space, RASS efficiently identifies and curates boundary-aligned prompts, enabling more effective and targeted mitigation of overrefusal. This approach not only provides a more precise and interpretable view of model safety decisions but also seamlessly extends to multilingual scenarios. We have explored the safety decision boundaries of various LLMs and construct the MORBench evaluation set to facilitate robust assessment of model safety and helpfulness across multiple languages. Code and datasets are available at https://github.com/Master-PLC/RASS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08627v2">NL in the Middle: Code Translation with LLMs and Intermediate Representations</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Studies show that large language models (LLMs) produce buggy code translations. One promising avenue to improve translation accuracy is through intermediate representations, which provide structured guidance for the translation process. We investigate whether LLM-based code translation can benefit from intermediate representations, specifically in the form of natural language (NL) summaries and abstract syntax trees (ASTs). Since prompt engineering greatly affects LLM performance, we consider several ways to integrate these representations, from one-shot to chain-of-thought (CoT) prompting. Using Open GPT4 8X7B and specialized StarCoder and CodeGen models on popular code translation benchmarks (CodeNet and AVATAR), we find that CoT with an intermediate NL summary performs best, with an increase of 13.8% and 6.7%, respectively, in successful translations for the best-performing model (Open GPT4 8X7B) compared to the zero-shot prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01658v2">CoPL: Collaborative Preference Learning for Personalizing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 19pages, 13 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Personalizing large language models (LLMs) is important for aligning outputs with diverse user preferences, yet existing methods struggle with flexibility and generalization. We propose CoPL (Collaborative Preference Learning), a graph-based collaborative filtering framework that models user-response relationships to enhance preference estimation, particularly in sparse annotation settings. By integrating a mixture of LoRA experts, CoPL efficiently fine-tunes LLMs while dynamically balancing shared and user-specific preferences. Additionally, an optimization-free adaptation strategy enables generalization to unseen users without fine-tuning. Experiments on UltraFeedback-P demonstrate that CoPL outperforms existing personalized reward models, effectively capturing both common and controversial preferences, making it a scalable solution for personalized LLM alignment. The code is available at https://github.com/ml-postech/CoPL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18216v2">Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. A key challenge is ensuring the LLMs have enough knowledge to address specific security requirements, such as information about existing attacks. For this, we propose an approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired by the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance while employing RAG and Self-Ranking, with an increase of up to 71%pt (on average 37%pt) and up to 43%pt (on average 6%pt) in the F2-Score for XSS and SQLi detection, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13978v1">LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology</a></div>
    <div class="paper-meta">
      📅 2025-09-17
      | 💬 Paper accepted in the proceedings of the ACM/IEEE Supercomputing Conference (SC). Cite it as Renan Souza, Timothy Poteet, Brian Etz, Daniel Rosendo, Amal Gueroudji, Woong Shin, Prasanna Balaprakash, and Rafael Ferreira da Silva. 2025. LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology. In SC Workshops (WORKS)
    </div>
    <details class="paper-abstract">
      Modern scientific discovery increasingly relies on workflows that process data across the Edge, Cloud, and High Performance Computing (HPC) continuum. Comprehensive and in-depth analyses of these data are critical for hypothesis validation, anomaly detection, reproducibility, and impactful findings. Although workflow provenance techniques support such analyses, at large scale, the provenance data become complex and difficult to analyze. Existing systems depend on custom scripts, structured queries, or static dashboards, limiting data interaction. In this work, we introduce an evaluation methodology, reference architecture, and open-source implementation that leverages interactive Large Language Model (LLM) agents for runtime data analysis. Our approach uses a lightweight, metadata-driven design that translates natural language into structured provenance queries. Evaluations across LLaMA, GPT, Gemini, and Claude, covering diverse query classes and a real-world chemistry workflow, show that modular design, prompt tuning, and Retrieval-Augmented Generation (RAG) enable accurate and insightful LLM agent responses beyond recorded provenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04537v3">Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09974v2">Analysing Safety Risks in LLMs Fine-Tuned with Pseudo-Malicious Cyber Security Data</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been used in many application domains, including cyber security. The application of LLMs in the cyber security domain presents significant opportunities, such as for enhancing threat analysis and malware detection, but it can also introduce critical risks and safety concerns, including potential personal data leakage and automated generation of new malware. Building on recent findings that fine-tuning LLMs with pseudo-malicious cyber security data significantly compromises their safety, this paper presents a comprehensive validation and extension of these safety risks using a different evaluation framework. We employ the garak red teaming framework with the OWASP Top 10 for LLM Applications to assess four open-source LLMs: Mistral 7B, Llama 3 8B, Gemma 2 9B, and DeepSeek R1 8B. Our evaluation confirms and extends previous findings, showing that fine-tuning reduces safety resilience across all tested LLMs (e.g., the failure rate of Mistral 7B against prompt injection increases from 9.1% to 68.7%). We further propose and evaluate a novel safety alignment approach that carefully rewords instruction-response pairs to include explicit safety precautions and ethical considerations. This work validates previous safety concerns through independent evaluation and introduces new methods for mitigating these risks, contributing towards the development of secure, trustworthy, and ethically aligned LLMs. This approach demonstrates that it is possible to maintain or even improve model safety while preserving technical utility, offering a practical path towards developing safer fine-tuning methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09334v3">CyberLLMInstruct: A Pseudo-malicious Dataset Revealing Safety-performance Trade-offs in Cyber Security LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into cyber security applications presents both opportunities and critical safety risks. We introduce CyberLLMInstruct, a dataset of 54,928 pseudo-malicious instruction-response pairs spanning cyber security tasks including malware analysis, phishing simulations, and zero-day vulnerabilities. Our comprehensive evaluation using seven open-source LLMs reveals a critical trade-off: while fine-tuning improves cyber security task performance (achieving up to 92.50% accuracy on CyberMetric), it severely compromises safety resilience across all tested models and attack vectors (e.g., Llama 3.1 8B's security score against prompt injection drops from 0.95 to 0.15). The dataset incorporates diverse sources including CTF challenges, academic papers, industry reports, and CVE databases to ensure comprehensive coverage of cyber security domains. Our findings highlight the unique challenges of securing LLMs in adversarial domains and establish the critical need for developing fine-tuning methodologies that balance performance gains with safety preservation in security-sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13942v1">Evaluating Classical Software Process Models as Coordination Mechanisms for LLM-Based Software Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-17
    </div>
    <details class="paper-abstract">
      [Background] Large Language Model (LLM)-based multi-agent systems (MAS) are transforming software development by enabling autonomous collaboration. Classical software processes such asWaterfall, V-Model, and Agile offer structured coordination patterns that can be repurposed to guide these agent interactions. [Aims] This study explores how traditional software development processes can be adapted as coordination scaffolds for LLM based MAS and examines their impact on code quality, cost, and productivity. [Method] We executed 11 diverse software projects under three process models and four GPT variants, totaling 132 runs. Each output was evaluated using standardized metrics for size (files, LOC), cost (execution time, token usage), and quality (code smells, AI- and human detected bugs). [Results] Both process model and LLM choice significantly affected system performance. Waterfall was most efficient, V-Model produced the most verbose code, and Agile achieved the highest code quality, albeit at higher computational cost. [Conclusions] Classical software processes can be effectively instantiated in LLM-based MAS, but each entails trade-offs across quality, cost, and adaptability. Process selection should reflect project goals, whether prioritizing efficiency, robustness, or structured validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13268v1">LLMs for energy and macronutrients estimation using only text data from 24-hour dietary recalls: a parameter-efficient fine-tuning experiment using a 10-shot prompt</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 https://github.com/rodrigo-carrillo/LLMs-Macronutrient-Estimation-NHANES-Adolescents
    </div>
    <details class="paper-abstract">
      BACKGROUND: Most artificial intelligence tools used to estimate nutritional content rely on image input. However, whether large language models (LLMs) can accurately predict nutritional values based solely on text descriptions of foods consumed remains unknown. If effective, this approach could enable simpler dietary monitoring without the need for photographs. METHODS: We used 24-hour dietary recalls from adolescents aged 12-19 years in the National Health and Nutrition Examination Survey (NHANES). An open-source quantized LLM was prompted using a 10-shot, chain-of-thought approach to estimate energy and five macronutrients based solely on text strings listing foods and their quantities. We then applied parameter-efficient fine-tuning (PEFT) to evaluate whether predictive accuracy improved. NHANES-calculated values served as the ground truth for energy, proteins, carbohydrates, total sugar, dietary fiber and total fat. RESULTS: In a pooled dataset of 11,281 adolescents (49.9% male, mean age 15.4 years), the vanilla LLM yielded poor predictions. The mean absolute error (MAE) was 652.08 for energy and the Lin's CCC <0.46 across endpoints. In contrast, the fine-tuned model performed substantially better, with energy MAEs ranging from 171.34 to 190.90 across subsets, and Lin's CCC exceeding 0.89 for all outcomes. CONCLUSIONS: When prompted using a chain-of-thought approach and fine-tuned with PEFT, open-source LLMs exposed solely to text input can accurately predict energy and macronutrient values from 24-hour dietary recalls. This approach holds promise for low-burden, text-based dietary monitoring tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13244v1">Evaluating LLM Alignment on Personality Inference from Real-World Interview Data</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in roles requiring nuanced psychological understanding, such as emotional support agents, counselors, and decision-making assistants. However, their ability to interpret human personality traits, a critical aspect of such applications, remains unexplored, particularly in ecologically valid conversational settings. While prior work has simulated LLM "personas" using discrete Big Five labels on social media data, the alignment of LLMs with continuous, ground-truth personality assessments derived from natural interactions is largely unexamined. To address this gap, we introduce a novel benchmark comprising semi-structured interview transcripts paired with validated continuous Big Five trait scores. Using this dataset, we systematically evaluate LLM performance across three paradigms: (1) zero-shot and chain-of-thought prompting with GPT-4.1 Mini, (2) LoRA-based fine-tuning applied to both RoBERTa and Meta-LLaMA architectures, and (3) regression using static embeddings from pretrained BERT and OpenAI's text-embedding-3-small. Our results reveal that all Pearson correlations between model predictions and ground-truth personality traits remain below 0.26, highlighting the limited alignment of current LLMs with validated psychological constructs. Chain-of-thought prompting offers minimal gains over zero-shot, suggesting that personality inference relies more on latent semantic representation than explicit reasoning. These findings underscore the challenges of aligning LLMs with complex human attributes and motivate future work on trait-specific prompting, context-aware modeling, and alignment-oriented fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13237v1">Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 18 pages, 9 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now solve multi-step problems by emitting extended chains of thought. During the process, they often re-derive the same intermediate steps across problems, inflating token usage and latency. This saturation of the context window leaves less capacity for exploration. We study a simple mechanism that converts recurring reasoning fragments into concise, reusable "behaviors" (name + instruction) via the model's own metacognitive analysis of prior traces. These behaviors are stored in a "behavior handbook" which supplies them to the model in-context at inference or distills them into parameters via supervised fine-tuning. This approach achieves improved test-time reasoning across three different settings - 1) Behavior-conditioned inference: Providing the LLM relevant behaviors in-context during reasoning reduces number of reasoning tokens by up to 46% while matching or improving baseline accuracy; 2) Behavior-guided self-improvement: Without any parameter updates, the model improves its own future reasoning by leveraging behaviors from its own past problem solving attempts. This yields up to 10% higher accuracy than a naive critique-and-revise baseline; and 3) Behavior-conditioned SFT: SFT on behavior-conditioned reasoning traces is more effective at converting non-reasoning models into reasoning models as compared to vanilla SFT. Together, these results indicate that turning slow derivations into fast procedural hints enables LLMs to remember how to reason, not just what to conclude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13201v1">Scaling Up Throughput-oriented LLM Inference Applications on Heterogeneous Opportunistic GPU Clusters with Pervasive Context Management</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The widespread growth in LLM developments increasingly demands more computational power from clusters than what they can supply. Traditional LLM applications inherently require huge static resource allocations, which force users to either wait in a long job queue and accept progress delay, or buy expensive hardware to fulfill their needs and exacerbate the demand-supply problem. However, not all LLM applications are latency-sensitive and can instead be executed in a throughput-oriented way. This throughput orientation allows a dynamic allocation that opportunistically pools available resources over time, avoiding both the long queue and expensive GPU purchases. Effectively utilizing opportunistic resources brings numerous challenges nevertheless. Our solution, pervasive context management, exploits the common computational context in LLM applications and provides mechanisms and policies that allow seamless context reuse on opportunistic resources. Our evaluation shows an LLM application with pervasive context management on opportunistic resources reduces its execution time by 98.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13179v1">Efficient Cold-Start Recommendation via BPE Token-Level Embedding Initialization with LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      The cold-start issue is the challenge when we talk about recommender systems, especially in the case when we do not have the past interaction data of new users or new items. Content-based features or hybrid solutions are common as conventional solutions, but they can only work in a sparse metadata environment with shallow patterns. In this paper, the efficient cold-start recommendation strategy is presented, which is based on the sub word-level representations by applying Byte Pair Encoding (BPE) tokenization and pre-trained Large Language Model (LLM) embedding in the initialization procedure. We obtain fine-grained token-level vectors that are aligned with the BPE vocabulary as opposed to using coarse-grained sentence embeddings. Together, these token embeddings can be used as dense semantic priors on unseen entities, making immediate recommendation performance possible without user-item interaction history. Our mechanism can be compared to collaborative filtering systems and tested over benchmark datasets with stringent cold-start assumptions. Experimental findings show that the given BPE-LLM method achieves higher Recall@k, NDCG@k, and Hit Rate measurements compared to the standard baseline and displays the same capability of sufficient computational performance. Furthermore, we demonstrate that using subword-aware embeddings yields better generalizability and is more interpretable, especially within a multilingual and sparse input setting. The practical application of token-level semantic initialization as a lightweight, but nevertheless effective extension to modern recommender systems in the zero-shot setting is indicated within this work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13175v1">More performant and scalable: Rethinking contrastive vision-language pre-training of radiology in the LLM era</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 MICCAI 2025
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) presents unprecedented opportunities to revolutionize medical contrastive vision-language pre-training. In this paper, we show how LLMs can facilitate large-scale supervised pre-training, thereby advancing vision-language alignment. We begin by demonstrate that modern LLMs can automatically extract diagnostic labels from radiology reports with remarkable precision (>96\% AUC in our experiments) without complex prompt engineering, enabling the creation of large-scale "silver-standard" datasets at a minimal cost (~\$3 for 50k CT image-report pairs). Further, we find that vision encoder trained on this "silver-standard" dataset achieves performance comparable to those trained on labels extracted by specialized BERT-based models, thereby democratizing the access to large-scale supervised pre-training. Building on this foundation, we proceed to reveal that supervised pre-training fundamentally improves contrastive vision-language alignment. Our approach achieves state-of-the-art performance using only a 3D ResNet-18 with vanilla CLIP training, including 83.8\% AUC for zero-shot diagnosis on CT-RATE, 77.3\% AUC on RAD-ChestCT, and substantial improvements in cross-modal retrieval (MAP@50=53.7\% for image-image, Recall@100=52.2\% for report-image). These results demonstrate the potential of utilizing LLMs to facilitate {\bf more performant and scalable} medical AI systems. Our code is avaiable at https://github.com/SadVoxel/More-performant-and-scalable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13154v1">LLM Hallucination Detection: A Fast Fourier Transform Method Based on Hidden Layer Temporal Signals</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Hallucination remains a critical barrier for deploying large language models (LLMs) in reliability-sensitive applications. Existing detection methods largely fall into two categories: factuality checking, which is fundamentally constrained by external knowledge coverage, and static hidden-state analysis, that fails to capture deviations in reasoning dynamics. As a result, their effectiveness and robustness remain limited. We propose HSAD (Hidden Signal Analysis-based Detection), a novel hallucination detection framework that models the temporal dynamics of hidden representations during autoregressive generation. HSAD constructs hidden-layer signals by sampling activations across layers, applies Fast Fourier Transform (FFT) to obtain frequency-domain representations, and extracts the strongest non-DC frequency component as spectral features. Furthermore, by leveraging the autoregressive nature of LLMs, HSAD identifies optimal observation points for effective and reliable detection. Across multiple benchmarks, including TruthfulQA, HSAD achieves over 10 percentage points improvement compared to prior state-of-the-art methods. By integrating reasoning-process modeling with frequency-domain analysis, HSAD establishes a new paradigm for robust hallucination detection in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13127v1">Empowering LLMs with Parameterized Skills for Adversarial Long-Horizon Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to IJCNN 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models(LLMs) have led to the development of LLM-based AI agents. A key challenge is the creation of agents that can effectively ground themselves in complex, adversarial long-horizon environments. Existing methods mainly focus on (1) using LLMs as policies to interact with the environment through generating low-level feasible actions, and (2) utilizing LLMs to generate high-level tasks or language guides to stimulate action generation. However, the former struggles to generate reliable actions, while the latter relies heavily on expert experience to translate high-level tasks into specific action sequences. To address these challenges, we introduce the Plan with Language, Act with Parameter (PLAP) planning framework that facilitates the grounding of LLM-based agents in long-horizon environments. The PLAP method comprises three key components: (1) a skill library containing environment-specific parameterized skills, (2) a skill planner powered by LLMs, and (3) a skill executor converting the parameterized skills into executable action sequences. We implement PLAP in MicroRTS, a long-horizon real-time strategy game that provides an unfamiliar and challenging environment for LLMs. The experimental results demonstrate the effectiveness of PLAP. In particular, GPT-4o-driven PLAP in a zero-shot setting outperforms 80% of baseline agents, and Qwen2-72B-driven PLAP, with carefully crafted few-shot examples, surpasses the top-tier scripted agent, CoacAI. Additionally, we design comprehensive evaluation metrics and test 6 closed-source and 2 open-source LLMs within the PLAP framework, ultimately releasing an LLM leaderboard ranking long-horizon skill planning ability. Our code is available at https://github.com/AI-Research-TeamX/PLAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18436v3">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can activate, or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13103v1">Accelerating Discovery: Rapid Literature Screening with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 This version of the manuscript has been submitted to Empirical Software Engieering Journal for consideration
    </div>
    <details class="paper-abstract">
      Background: Conducting Multi Vocal Literature Reviews (MVLRs) is often time and effort-intensive. Researchers must review and filter a large number of unstructured sources, which frequently contain sparse information and are unlikely to be included in the final study. Our experience conducting an MVLR on Context-Aware Software Systems (CASS) Testing in the avionics domain exemplified this challenge, with over 8,000 highly heterogeneous documents requiring review. Therefore, we developed a Large Language Model (LLM) assistant to support the search and filtering of documents. Aims: To develop and validate an LLM based tool that can support researchers in performing the search and filtering of documents for an MVLR without compromising the rigor of the research protocol. Method: We applied sound engineering practices to develop an on-premises LLM-based tool incorporating Retrieval Augmented Generation (RAG) to process candidate sources. Progress towards the aim was quantified using the Positive Percent Agreement (PPA) as the primary metric to ensure the performance of the LLM based tool. Convenience sampling, supported by human judgment and statistical sampling, were used to verify and validate the tool's quality-in-use. Results: The tool currently demonstrates a PPA agreement with human researchers of 90% for sources that are not relevant to the study. Development details are shared to support domain-specific adaptation of the tool. Conclusions: Using LLM-based tools to support academic researchers in rigorous MVLR is feasible. These tools can free valuable time for higher-level, abstract tasks. However, researcher participation remains essential to ensure that the tool supports thorough research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21740v2">Counterfactual Simulatability of LLM Explanations for Generation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13055v1">Automating Code Generation for Semiconductor Equipment Control from Developer Utterances with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Semiconductors form the backbone of modern electronics, with their manufacturing and testing relying on highly specialized equipment and domain-specific programming languages. Equipment languages such as the Algorithmic Pattern Generator (ALPG) are critical for precise hardware control but are challenging to program due to their low-level syntax and steep learning curve. While large language models (LLMs) have shown promise in generating high-level code from natural language, their effectiveness on low-level equipment languages remains limited. To address this, we propose Progressive Knowledge Enhancement (PKE), a novel multi-stage prompting framework that progressively extracts and activates the latent knowledge within LLMs, guiding them from simple to complex examples without extensive fine-tuning. Empirical evaluation on an industrial ALPG dataset shows that PKE significantly outperforms standard prompting and surpasses state-of-the-art methods in generating correct ALPG code, achieving 11.1\% and 15.2\% higher exact match scores compared to the second-best technique. Further analysis of individual components confirms that progressive knowledge extraction based on difficulty enhances accuracy. Our study offer a practical approach to boosting LLM capabilities for specialized low-level programming, supporting greater productivity in semiconductor software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08045v3">Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted at EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      A large number of studies rely on closed-style multiple-choice surveys to evaluate cultural alignment in Large Language Models (LLMs). In this work, we challenge this constrained evaluation paradigm and explore more realistic, unconstrained approaches. Using the World Values Survey (WVS) and Hofstede Cultural Dimensions as case studies, we demonstrate that LLMs exhibit stronger cultural alignment in less constrained settings, where responses are not forced. Additionally, we show that even minor changes, such as reordering survey choices, lead to inconsistent outputs, exposing the limitations of closed-style evaluations. Our findings advocate for more robust and flexible evaluation frameworks that focus on specific cultural proxies, encouraging more nuanced and accurate assessments of cultural alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16467v2">Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted at EMNLP Main 2025
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs) infer user's demographic information from subtle cues in the conversation -- a phenomenon called implicit personalization. Prior work has shown that such inferences can lead to lower quality responses for users assumed to be from minority groups, even when no demographic information is explicitly provided. In this work, we systematically explore how LLMs respond to stereotypical cues using controlled synthetic conversations, by analyzing the models' latent user representations through both model internals and generated answers to targeted user questions. Our findings reveal that LLMs do infer demographic attributes based on these stereotypical signals, which for a number of groups even persists when the user explicitly identifies with a different demographic group. Finally, we show that this form of stereotype-driven implicit personalization can be effectively mitigated by intervening on the model's internal representations using a trained linear probe to steer them toward the explicitly stated identity. Our results highlight the need for greater transparency and control in how LLMs represent user identity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13021v1">xOffense: An AI-driven autonomous penetration testing framework with offensive knowledge-enhanced LLMs and multi agent systems</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 17 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This work introduces xOffense, an AI-driven, multi-agent penetration testing framework that shifts the process from labor-intensive, expert-driven manual efforts to fully automated, machine-executable workflows capable of scaling seamlessly with computational infrastructure. At its core, xOffense leverages a fine-tuned, mid-scale open-source LLM (Qwen3-32B) to drive reasoning and decision-making in penetration testing. The framework assigns specialized agents to reconnaissance, vulnerability scanning, and exploitation, with an orchestration layer ensuring seamless coordination across phases. Fine-tuning on Chain-of-Thought penetration testing data further enables the model to generate precise tool commands and perform consistent multi-step reasoning. We evaluate xOffense on two rigorous benchmarks: AutoPenBench and AI-Pentest-Benchmark. The results demonstrate that xOffense consistently outperforms contemporary methods, achieving a sub-task completion rate of 79.17%, decisively surpassing leading systems such as VulnBot and PentestGPT. These findings highlight the potential of domain-adapted mid-scale LLMs, when embedded within structured multi-agent orchestration, to deliver superior, cost-efficient, and reproducible solutions for autonomous penetration testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19894v3">TRANSAGENT: An LLM-Based Multi-Agent System for Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Code translation converts code from one programming language to another while maintaining its original functionality, which is crucial for software migration, system refactoring, and cross-platform development. Traditional rule-based methods rely on manually-written rules, which can be time-consuming and often result in less readable code. To overcome this, learning-based methods have been developed, leveraging parallel data to train models for automated code translation. More recently, the advance of Large Language Models (LLMs) further boosts learning-based code translation. Although promising, LLM-translated program still suffers from diverse quality issues (e.g., syntax errors and semantic errors). In particular, it can be challenging for LLMs to self-debug these errors when simply provided with the corresponding error messages. In this work, we propose a novel LLM-based multi-agent system TRANSAGENT, which enhances LLM-based code translation by fixing the syntax errors and semantic errors with the synergy between four LLM-based agents, including Initial Code Translator, Syntax Error Fixer, Code Aligner, and Semantic Error Fixer. The main insight of TRANSAGENT is to first localize the error code block in the target program based on the execution alignment between the target and source program, which can narrow down the fixing space and thus lower down the fixing difficulties. To evaluate TRANSAGENT, we first construct a new benchmark from recent programming tasks to mitigate the potential data leakage issue. On our benchmark, TRANSAGENT outperforms the latest LLM-based code translation technique UniTrans in both translation effectiveness and efficiency; additionally, our evaluation on different LLMs show the generalization of TRANSAGENT and our ablation study shows the contribution of each agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12067v3">TokenSkip: Controllable Chain-of-Thought Compression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 (Long Paper), camera-ready version
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop. We release our code and checkpoints in https://github.com/hemingkx/TokenSkip.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16408v2">From Surveys to Narratives: Rethinking Cultural Value Adaptation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Adapting cultural values in Large Language Models (LLMs) presents significant challenges, particularly due to biases and limited training data. Prior work primarily aligns LLMs with different cultural values using World Values Survey (WVS) data. However, it remains unclear whether this approach effectively captures cultural nuances or produces distinct cultural representations for various downstream tasks. In this paper, we systematically investigate WVS-based training for cultural value adaptation and find that relying solely on survey data can homogenize cultural norms and interfere with factual knowledge. To investigate these issues, we augment WVS with encyclopedic and scenario-based cultural narratives from Wikipedia and NormAd. While these narratives may have variable effects on downstream tasks, they consistently improve cultural distinctiveness than survey data alone. Our work highlights the inherent complexity of aligning cultural values with the goal of guiding task-specific behavior. We release our code at https://github.com/faridlazuarda/from-surveys-to-narratives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12769v3">How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12961v1">Do LLMs Understand Wine Descriptors Across Cultures? A Benchmark for Cultural Adaptations of Wine Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have opened the door to culture-aware language tasks. We introduce the novel problem of adapting wine reviews across Chinese and English, which goes beyond literal translation by incorporating regional taste preferences and culture-specific flavor descriptors. In a case study on cross-cultural wine review adaptation, we compile the first parallel corpus of professional reviews, containing 8k Chinese and 16k Anglophone reviews. We benchmark both neural-machine-translation baselines and state-of-the-art LLMs with automatic metrics and human evaluation. For the latter, we propose three culture-oriented criteria -- Cultural Proximity, Cultural Neutrality, and Cultural Genuineness -- to assess how naturally a translated review resonates with target-culture readers. Our analysis shows that current models struggle to capture cultural nuances, especially in translating wine descriptions across different cultures. This highlights the challenges and limitations of translation models in handling cultural content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08523v3">InferLog: Accelerating LLM Inference for Online Log Parsing via ICL-oriented Prefix Caching</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted by ICSE '26 (The 48th IEEE/ACM International Conference on Software Engineering)
    </div>
    <details class="paper-abstract">
      Modern software systems generate massive volumes of runtime logs, necessitating efficient and accurate log parsing to enable critical downstream tasks such as anomaly detection and root cause analysis. Recently, large language models (LLMs) have achieved advanced accuracy on log parsing, but their deployment in production environments faces two major limitations: (1) the privacy risks associated with commercial LLMs, driving the adoption of local deployment, and (2) the stringent latency and throughput requirements imposed by high-volume log streams, which existing LLM-based parsers fail to meet. Although recent efforts have reduced the number of LLM queries, they overlook the high latency of the LLM invocations, where concurrent log parsing requests can cause serve performance degradation of LLM inference system. In this study, we present InferLog, the first LLM inference optimization method for online log parsing. Our key insight is that the inference efficiency emerges as the vital bottleneck in LLM-based online log parsing, rather than parsing accuracy. InferLog accelerates inference by designing (1) A Prefix-aware ICL Refinement policy to refine the examples and permutation of in-context learning to improve the prefix caching efficiency. (2) A rapid and task-specific configuration tuning pipeline based on meta-learning to find the optimal LLM scheduling-related configuration for dynamic log parsing workloads. The experimental results based on Loghub dataset and vLLM demonstrate that InferLog significantly outperforms existing inference optimization methods and markedly accelerates the state-of-the-art LLM-based log parser without compromising parsing accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12914v1">Stochastic Streets: A Walk Through Random LLM Address Generation in four European Cities</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of solving complex math problems or answer difficult questions on almost any topic, but can they generate random street addresses for European cities?
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12892v1">Conan-Embedding-v2: Training an LLM from Scratch for Text Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 Oral
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated excellent performance in text embedding tasks. Previous work usually use LoRA to fine-tune existing LLMs, which are limited by the data and training gap between LLMs and embedding models. In this work, we introduce Conan-embedding-v2, a new 1.4B-parameter LLM trained from scratch and fine-tuned as a text embedder. First, we add news data and multilingual pairs for LLM pretraining to bridge the data gap. Based on this, we propose a cross-lingual retrieval dataset that enables the LLM to better integrate embeddings across different languages. Second, whereas LLMs use a causal mask with token-level loss, embedding models use a bidirectional mask with sentence-level loss. This training gap makes full fine-tuning less effective than LoRA. We introduce a soft-masking mechanism to gradually transition between these two types of masks, enabling the model to learn more comprehensive representations. Based on this, we propose a dynamic hard negative mining method that exposes the model to more difficult negative examples throughout the training process. Being intuitive and effective, with only approximately 1.4B parameters, Conan-embedding-v2 achieves SOTA performance on both the Massive Text Embedding Benchmark (MTEB) and Chinese MTEB (May 19, 2025).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12886v1">The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Estimating the difficulty of input questions as perceived by large language models (LLMs) is essential for accurate performance evaluation and adaptive inference. Existing methods typically rely on repeated response sampling, auxiliary models, or fine-tuning the target model itself, which may incur substantial computational costs or compromise generality. In this paper, we propose a novel approach for difficulty estimation that leverages only the hidden representations produced by the target LLM. We model the token-level generation process as a Markov chain and define a value function to estimate the expected output quality given any hidden state. This allows for efficient and accurate difficulty estimation based solely on the initial hidden state, without generating any output tokens. Extensive experiments across both textual and multimodal tasks demonstrate that our method consistently outperforms existing baselines in difficulty estimation. Moreover, we apply our difficulty estimates to guide adaptive reasoning strategies, including Self-Consistency, Best-of-N, and Self-Refine, achieving higher inference efficiency with fewer generated tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01129v5">Emphasising Structured Information: Integrating Abstract Meaning Representation into LLMs for Enhanced Open-Domain Dialogue Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Automatic open-domain dialogue evaluation has attracted increasing attention, yet remains challenging due to the complexity of assessing response appropriateness. Traditional evaluation metrics, typically trained with true positive and randomly selected negative responses, tend to assign higher scores to responses that share greater content similarity with contexts. However, adversarial negative responses, despite possessing high lexical overlap with contexts, can be semantically incongruous. Consequently, existing metrics struggle to effectively evaluate such responses, resulting in low correlations with human judgments. While recent studies have demonstrated the effectiveness of Large Language Models (LLMs) for open-domain dialogue evaluation, they still face challenges in handling adversarial negative examples. We propose a novel evaluation framework that integrates Abstract Meaning Representation (AMR) enhanced domain-specific language models (SLMs) with LLMs. Our SLMs explicitly incorporate AMR graph information through a gating mechanism for enhanced semantic representation learning, while both SLM predictions and AMR knowledge are integrated into LLM prompts for robust evaluation. Extensive experiments on open-domain dialogue evaluation tasks demonstrate the superiority of our method compared to state-of-the-art baselines. Our comprehensive ablation studies reveal that AMR graph information contributes substantially more to performance improvements. Our framework achieves strong correlations with human judgments across multiple datasets, establishing a new benchmark for dialogue evaluation. Our code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17962v7">Crafting Customisable Characters with LLMs: A Persona-Driven Role-Playing Agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable ability to comprehend instructions and generate human-like text, enabling sophisticated agent simulation beyond basic behavior replication. However, the potential for creating freely customisable characters remains underexplored. We introduce the Customisable Conversation Agent Framework, which employs LLMs to simulate real-world characters through personalised characteristic feature injection, enabling diverse character creation according to user preferences. We propose the SimsConv dataset, comprising 68 customised characters and 13,971 multi-turn role-playing dialogues across 1,360 real-world scenes. Characters are initially customised using pre-defined elements (career, aspiration, traits, skills), then expanded through personal and social profiles. Building on this, we present SimsChat, a freely customisable role-playing agent incorporating various realistic settings and topic-specified character interactions. Experimental results on both SimsConv and WikiRoleEval datasets demonstrate SimsChat's superior performance in maintaining character consistency, knowledge accuracy, and appropriate question rejection compared to existing models. Our framework provides valuable insights for developing more accurate and customisable human simulacra. Our data and code are publicly available at https://github.com/Bernard-Yang/SimsChat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12810v1">H$^2$R: Hierarchical Hindsight Reflection for Multi-Task LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents have shown strong potential in multi-task scenarios, owing to their ability to transfer knowledge across diverse tasks. However, existing approaches often treat prior experiences and knowledge as monolithic units, leading to inefficient and coarse-grained knowledge transfer. In this work, we propose a novel hierarchical memory architecture that enables fine-grained knowledge transfer by decoupling high-level planning memory from low-level execution memory. To construct and refine these hierarchical memories, we introduce Hierarchical Hindsight Reflection (H$^2$R), a mechanism that distills reusable and hierarchical knowledge from past agent-environment interactions. At test time, H$^2$R performs retrievals of high-level and low-level memories separately, allowing LLM-based agents to efficiently access and utilize task-relevant knowledge for new tasks.Experimental results across two benchmarks demonstrate that H$^2$R can improve generalization and decision-making performance, outperforming prior baselines such as Expel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05179v3">Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled strong reasoning capabilities through Chain-of-Thought (CoT) prompting, which elicits step-by-step problem solving, but often at the cost of excessive verbosity in intermediate outputs, leading to increased computational overhead. We propose Sketch-of-Thought (SoT), a prompting framework that integrates cognitively inspired reasoning paradigms with linguistic constraints to reduce token usage while preserving reasoning accuracy. SoT is designed as a flexible, modular approach and is instantiated with three paradigms--Conceptual Chaining, Chunked Symbolism, and Expert Lexicons--each tailored to distinct reasoning tasks and selected dynamically at test-time by a lightweight routing model. Across 18 reasoning datasets spanning multiple domains, languages, and modalities, SoT achieves token reductions of up to 84% with minimal accuracy loss. In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12798v1">LLM-Based Approach for Enhancing Maintainability of Automotive Architectures</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      There are many bottlenecks that decrease the flexibility of automotive systems, making their long-term maintenance, as well as updates and extensions in later lifecycle phases increasingly difficult, mainly due to long re-engineering, standardization, and compliance procedures, as well as heterogeneity and numerosity of devices and underlying software components involved. In this paper, we explore the potential of Large Language Models (LLMs) when it comes to the automation of tasks and processes that aim to increase the flexibility of automotive systems. Three case studies towards achieving this goal are considered as outcomes of early-stage research: 1) updates, hardware abstraction, and compliance, 2) interface compatibility checking, and 3) architecture modification suggestions. For proof-of-concept implementation, we rely on OpenAI's GPT-4o model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20999v3">LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17711v3">Enhancing LLM-Based Social Bot via an Adversarial Learning Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Developing Large Language Model (LLM) agents that exhibit human-like behavior, encompassing not only individual heterogeneity rooted in unique user profiles but also adaptive response to socially connected neighbors, is a significant research challenge. Social media platforms, with their diverse user data and explicit social structures, provide an ideal testbed for such investigations. This paper introduces EvoBot, an \textbf{Evo}lving LLM-based social \textbf{Bot} that significantly enhances human-like generative capabilities through a novel adversarial learning framework. EvoBot is initialized by Supervised Fine-Tuning (SFT) on representative data from social media and then iteratively refines its generation of sophisticated, human-like content via Direct Preference Optimization (DPO). This refinement is guided by feedback from a co-adapting \textbf{Detector} which concurrently improves its ability to distinguish EvoBot from humans, thereby creating an increasingly challenging learning environment for EvoBot. Experiments demonstrate that EvoBot generates content aligned with diverse user profiles, increasingly bypassing the co-adapting Detector through human-like expression. Moreover, it exhibits strong social responsiveness, more accurately modeling real-world opinion dynamics and information spread in multi-agent simulations. The framework also yields a more robust Detector, underscoring its broader utility for both advanced agent development and related detection tasks. The code is available at https://github.com/kfq20/EvoBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12750v1">What Makes a Good Generated Image? Investigating Human and Multimodal LLM Image Preference Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 7 pages, 9 figures, 3 tables; appendix 16 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Automated evaluation of generative text-to-image models remains a challenging problem. Recent works have proposed using multimodal LLMs to judge the quality of images, but these works offer little insight into how multimodal LLMs make use of concepts relevant to humans, such as image style or composition, to generate their overall assessment. In this work, we study what attributes of an image--specifically aesthetics, lack of artifacts, anatomical accuracy, compositional correctness, object adherence, and style--are important for both LLMs and humans to make judgments on image quality. We first curate a dataset of human preferences using synthetically generated image pairs. We use inter-task correlation between each pair of image quality attributes to understand which attributes are related in making human judgments. Repeating the same analysis with LLMs, we find that the relationships between image quality attributes are much weaker. Finally, we study individual image quality attributes by generating synthetic datasets with a high degree of control for each axis. Humans are able to easily judge the quality of an image with respect to all of the specific image quality attributes (e.g. high vs. low aesthetic image), however we find that some attributes, such as anatomical accuracy, are much more difficult for multimodal LLMs to learn to judge. Taken together, these findings reveal interesting differences between how humans and multimodal LLMs perceive images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12743v1">Zero-shot Graph Reasoning via Retrieval Augmented Framework with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      We propose a new, training-free method, Graph Reasoning via Retrieval Augmented Framework (GRRAF), that harnesses retrieval-augmented generation (RAG) alongside the code-generation capabilities of large language models (LLMs) to address a wide range of graph reasoning tasks. In GRRAF, the target graph is stored in a graph database, and the LLM is prompted to generate executable code queries that retrieve the necessary information. This approach circumvents the limitations of existing methods that require extensive finetuning or depend on predefined algorithms, and it incorporates an error feedback loop with a time-out mechanism to ensure both correctness and efficiency. Experimental evaluations on the GraphInstruct dataset reveal that GRRAF achieves 100% accuracy on most graph reasoning tasks, including cycle detection, bipartite graph checks, shortest path computation, and maximum flow, while maintaining consistent token costs regardless of graph sizes. Imperfect but still very high performance is observed on subgraph matching. Notably, GRRAF scales effectively to large graphs with up to 10,000 nodes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19345v2">PatentScore: Multi-dimensional Evaluation of LLM-Generated Patent Claims</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      High-stakes texts such as patent claims, medical records, and technical reports are structurally complex and demand a high degree of reliability and precision. While large language models (LLMs) have recently been applied to automate their generation in high-stakes domains, reliably evaluating such outputs remains a major challenge. Conventional natural language generation (NLG) metrics are effective for generic documents but fail to capture the structural and legal characteristics essential to evaluating complex high-stakes documents. To address this gap, we propose PatentScore, a multi-dimensional evaluation framework specifically designed for one of the most intricate and rigorous domains, patent claims. PatentScore integrates hierarchical decomposition of claim elements, validation patterns grounded in legal and technical standards, and scoring across structural, semantic, and legal dimensions. In experiments on our dataset which consists of 400 Claim1, PatentScore achieved the highest correlation with expert annotations ($r = 0.819$), significantly outperforming widely used NLG metrics. This work establishes a new standard for evaluating LLM-generated patent claims, providing a solid foundation for research on patent generation and validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10259v2">Temporal-Aware GPU Resource Allocation for Distributed LLM Inference via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 17 pages, 12 figures
    </div>
    <details class="paper-abstract">
      The rapid growth of large language model (LLM) services imposes increasing demands on distributed GPU inference infrastructure. Most existing scheduling systems follow a reactive paradigm, relying solely on the current system state to make decisions, without considering how task demand and resource availability evolve over time. This lack of temporal awareness in reactive approaches leads to inefficient GPU utilization, high task migration overhead, and poor system responsiveness under dynamic workloads. In this work, we identify the fundamental limitations of these instantaneous-state-only scheduling approaches and propose Temporal Optimal Resource scheduling via Two-layer Architecture (TORTA). TORTA introduces a spatiotemporal scheduling framework that captures both long-term workload patterns and short-term execution constraints. It adopts a two-layer design: a macro-level scheduler leverages reinforcement learning and optimal transport to coordinate inter-region task distribution, while a micro-level allocator refines task-to-server assignments within each region to reduce latency and switching costs. Experimental results across multiple network topologies show that TORTA reduces average inference response time by up to 15\%, improves load balance by approximately 4-5\%, and cuts total operational cost by 10-20\% compared to state-of-the-art baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18530v3">IMPROVE: Iterative Model Pipeline Refinement and Optimization Leveraging LLM Experts</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have emerged as a promising solution to automate the workflow of machine learning, but most existing methods share a common limitation: they attempt to optimize entire pipelines in a single step before evaluation, making it difficult to attribute improvements to specific changes. This lack of granularity leads to unstable optimization and slower convergence, limiting their effectiveness. To address this, we introduce Iterative Refinement, a novel strategy for LLM-driven ML pipeline design inspired by how human ML experts iteratively refine models, focusing on one component at a time rather than making sweeping changes all at once. By systematically updating individual components based on real training feedback, Iterative Refinement improves overall model performance. We also provide some theoretical edvience of the superior properties of this Iterative Refinement. Further, we implement this strategy in IMPROVE, an end-to-end LLM agent framework for automating and optimizing object classification pipelines. Through extensive evaluations across datasets of varying sizes and domains, we demonstrate that Iterative Refinement enables IMPROVE to consistently achieve better performance over existing zero-shot LLM-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12678v1">Instance-level Randomization: Toward More Stable LLM Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-09-16
      | 💬 Accepted by Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Evaluations of large language models (LLMs) suffer from instability, where small changes of random factors such as few-shot examples can lead to drastic fluctuations of scores and even model rankings. Moreover, different LLMs can have different preferences for a certain setting of random factors. As a result, using a fixed setting of random factors, which is often adopted as the paradigm of current evaluations, can lead to potential unfair comparisons between LLMs. To mitigate the volatility of evaluations, we first theoretically analyze the sources of variance induced by changes in random factors. Targeting these specific sources, we then propose the instance-level randomization (ILR) method to reduce variance and enhance fairness in model comparisons. Instead of using a fixed setting across the whole benchmark in a single experiment, we randomize all factors that affect evaluation scores for every single instance, run multiple experiments and report the averaged score. Theoretical analyses and empirical results demonstrate that ILR can reduce the variance and unfair comparisons caused by random factors, as well as achieve similar robustness level with less than half computational cost compared with previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12672v1">Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content</a></div>
    <div class="paper-meta">
      📅 2025-09-16
    </div>
    <details class="paper-abstract">
      The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.
    </details>
</div>
