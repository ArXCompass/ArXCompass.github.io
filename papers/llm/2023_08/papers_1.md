# llm - 2023_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.09162v3">Unveiling Gender Bias in Terms of Profession Across LLMs: Analyzing and Addressing Sociological Implications</a></div>
    <div class="paper-meta">
      📅 2023-08-31
    </div>
    <details class="paper-abstract">
      Gender bias in artificial intelligence (AI) and natural language processing has garnered significant attention due to its potential impact on societal perceptions and biases. This research paper aims to analyze gender bias in Large Language Models (LLMs) with a focus on multiple comparisons between GPT-2 and GPT-3.5, some prominent language models, to better understand its implications. Through a comprehensive literature review, the study examines existing research on gender bias in AI language models and identifies gaps in the current knowledge. The methodology involves collecting and preprocessing data from GPT-2 and GPT-3.5, and employing in-depth quantitative analysis techniques to evaluate gender bias in the generated text. The findings shed light on gendered word associations, language usage, and biased narratives present in the outputs of these Large Language Models. The discussion explores the ethical implications of gender bias and its potential consequences on social perceptions and marginalized communities. Additionally, the paper presents strategies for reducing gender bias in LLMs, including algorithmic approaches and data augmentation techniques. The research highlights the importance of interdisciplinary collaborations and the role of sociological studies in mitigating gender bias in AI models. By addressing these issues, we can pave the way for more inclusive and unbiased AI systems that have a positive impact on society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.16753v1">Context Aware Query Rewriting for Text Rankers using LLM</a></div>
    <div class="paper-meta">
      📅 2023-08-31
    </div>
    <details class="paper-abstract">
      Query rewriting refers to an established family of approaches that are applied to underspecified and ambiguous queries to overcome the vocabulary mismatch problem in document ranking. Queries are typically rewritten during query processing time for better query modelling for the downstream ranker. With the advent of large-language models (LLMs), there have been initial investigations into using generative approaches to generate pseudo documents to tackle this inherent vocabulary gap. In this work, we analyze the utility of LLMs for improved query rewriting for text ranking tasks. We find that there are two inherent limitations of using LLMs as query re-writers -- concept drift when using only queries as prompts and large inference costs during query processing. We adopt a simple, yet surprisingly effective, approach called context aware query rewriting (CAR) to leverage the benefits of LLMs for query understanding. Firstly, we rewrite ambiguous training queries by context-aware prompting of LLMs, where we use only relevant documents as context.Unlike existing approaches, we use LLM-based query rewriting only during the training phase. Eventually, a ranker is fine-tuned on the rewritten queries instead of the original queries during training. In our extensive experiments, we find that fine-tuning a ranker using re-written queries offers a significant improvement of up to 33% on the passage ranking task and up to 28% on the document ranking task when compared to the baseline performance of using original queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.16369v1">SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills</a></div>
    <div class="paper-meta">
      📅 2023-08-31
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference consists of two distinct phases - prefill phase which processes the input prompt and decode phase which generates output tokens autoregressively. While the prefill phase effectively saturates GPU compute at small batch sizes, the decode phase results in low compute utilization as it generates one token at a time per request. The varying prefill and decode times also lead to imbalance across micro-batches when using pipeline parallelism, resulting in further inefficiency due to bubbles. We present SARATHI to address these challenges. SARATHI employs chunked-prefills, which splits a prefill request into equal sized chunks, and decode-maximal batching, which constructs a batch using a single prefill chunk and populates the remaining slots with decodes. During inference, the prefill chunk saturates GPU compute, while the decode requests 'piggyback' and cost up to an order of magnitude less compared to a decode-only batch. Chunked-prefills allows constructing multiple decode-maximal batches from a single prefill request, maximizing coverage of decodes that can piggyback. Furthermore, the uniform compute design of these batches ameliorates the imbalance between micro-batches, significantly reducing pipeline bubbles. Our techniques yield significant improvements in inference performance across models and hardware. For the LLaMA-13B model on A6000 GPU, SARATHI improves decode throughput by up to 10x, and accelerates end-to-end throughput by up to 1.33x. For LLaMa-33B on A100 GPU, we achieve 1.25x higher end-to-end-throughput and up to 4.25x higher decode throughput. When used with pipeline parallelism on GPT-3, SARATHI reduces bubbles by 6.29x, resulting in an end-to-end throughput improvement of 1.91x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.15214v2">FurChat: An Embodied Conversational Agent using LLMs, Combining Open and Closed-Domain Dialogue with Facial Expressions</a></div>
    <div class="paper-meta">
      📅 2023-08-30
      | 💬 5 pages, 2 figures, Accepted at SIGDIAL 2023 (24th Meeting of the Special Interest Group on Discourse and Dialogue), for the demo video, see https://youtu.be/fwtUl1kl22s
    </div>
    <details class="paper-abstract">
      We demonstrate an embodied conversational agent that can function as a receptionist and generate a mixture of open and closed-domain dialogue along with facial expressions, by using a large language model (LLM) to develop an engaging conversation. We deployed the system onto a Furhat robot, which is highly expressive and capable of using both verbal and nonverbal cues during interaction. The system was designed specifically for the National Robotarium to interact with visitors through natural conversations, providing them with information about the facilities, research, news, upcoming events, etc. The system utilises the state-of-the-art GPT-3.5 model to generate such information along with domain-general conversations and facial expressions based on prompt engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.10235v4">Assessing Hidden Risks of LLMs: An Empirical Study on Robustness, Consistency, and Credibility</a></div>
    <div class="paper-meta">
      📅 2023-08-30
    </div>
    <details class="paper-abstract">
      The recent popularity of large language models (LLMs) has brought a significant impact to boundless fields, particularly through their open-ended ecosystem such as the APIs, open-sourced models, and plugins. However, with their widespread deployment, there is a general lack of research that thoroughly discusses and analyzes the potential risks concealed. In that case, we intend to conduct a preliminary but pioneering study covering the robustness, consistency, and credibility of LLMs systems. With most of the related literature in the era of LLM uncharted, we propose an automated workflow that copes with an upscaled number of queries/responses. Overall, we conduct over a million queries to the mainstream LLMs including ChatGPT, LLaMA, and OPT. Core to our workflow consists of a data primitive, followed by an automated interpreter that evaluates these LLMs under different adversarial metrical systems. As a result, we draw several, and perhaps unfortunate, conclusions that are quite uncommon from this trendy community. Briefly, they are: (i)-the minor but inevitable error occurrence in the user-generated query input may, by chance, cause the LLM to respond unexpectedly; (ii)-LLMs possess poor consistency when processing semantically similar query input. In addition, as a side finding, we find that ChatGPT is still capable to yield the correct answer even when the input is polluted at an extreme level. While this phenomenon demonstrates the powerful memorization of the LLMs, it raises serious concerns about using such data for LLM-involved evaluation in academic development. To deal with it, we propose a novel index associated with a dataset that roughly decides the feasibility of using such data for LLM-involved evaluation. Extensive empirical studies are tagged to support the aforementioned claims.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.15231v1">Multi-party Goal Tracking with LLMs: Comparing Pre-training, Fine-tuning, and Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2023-08-29
      | 💬 Accepted and will appear in the Proceedings of SIGdial 2023
    </div>
    <details class="paper-abstract">
      This paper evaluates the extent to which current Large Language Models (LLMs) can capture task-oriented multi-party conversations (MPCs). We have recorded and transcribed 29 MPCs between patients, their companions, and a social robot in a hospital. We then annotated this corpus for multi-party goal-tracking and intent-slot recognition. People share goals, answer each other's goals, and provide other people's goals in MPCs - none of which occur in dyadic interactions. To understand user goals in MPCs, we compared three methods in zero-shot and few-shot settings: we fine-tuned T5, created pre-training tasks to train DialogLM using LED, and employed prompt engineering techniques with GPT-3.5-turbo, to determine which approach can complete this novel task with limited data. GPT-3.5-turbo significantly outperformed the others in a few-shot setting. The `reasoning' style prompt, when given 7% of the corpus as example annotated conversations, was the best performing method. It correctly annotated 62.32% of the goal tracking MPCs, and 69.57% of the intent-slot recognition MPCs. A `story' style prompt increased model hallucination, which could be detrimental if deployed in safety-critical settings. We conclude that multi-party conversations still challenge state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.14972v1">LLM-Based Human-Robot Collaboration Framework for Manipulation Tasks</a></div>
    <div class="paper-meta">
      📅 2023-08-29
      | 💬 IEEE MHS 2023
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to enhance autonomous robotic manipulation using the Large Language Model (LLM) for logical inference, converting high-level language commands into sequences of executable motion functions. The proposed system combines the advantage of LLM with YOLO-based environmental perception to enable robots to autonomously make reasonable decisions and task planning based on the given commands. Additionally, to address the potential inaccuracies or illogical actions arising from LLM, a combination of teleoperation and Dynamic Movement Primitives (DMP) is employed for action correction. This integration aims to improve the practicality and generalizability of the LLM-based human-robot collaboration system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.17118v2">Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time</a></div>
    <div class="paper-meta">
      📅 2023-08-28
    </div>
    <details class="paper-abstract">
      Large language models(LLMs) have sparked a new wave of exciting AI applications. Hosting these models at scale requires significant memory resources. One crucial memory bottleneck for the deployment stems from the context window. It is commonly recognized that model weights are memory hungry; however, the size of key-value embedding stored during the generation process (KV cache) can easily surpass the model size. The enormous size of the KV cache puts constraints on the inference batch size, which is crucial for high throughput inference workload. Inspired by an interesting observation of the attention scores, we hypothesize the persistence of importance: only pivotal tokens, which had a substantial influence at one step, will significantly influence future generations. Based on our empirical verification and theoretical analysis around this hypothesis, we propose Scissorhands, a system that maintains the memory usage of the KV cache at a fixed budget without finetuning the model. In essence, Scissorhands manages the KV cache by storing the pivotal tokens with a higher probability. We validate that Scissorhands reduces the inference memory usage of the KV cache by up to 5X without compromising model quality. We further demonstrate that Scissorhands can be combined with 4-bit quantization, traditionally used to compress model weights, to achieve up to 20X compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.01358v3">Challenging the appearance of machine intelligence: Cognitive bias in LLMs and Best Practices for Adoption</a></div>
    <div class="paper-meta">
      📅 2023-08-28
    </div>
    <details class="paper-abstract">
      Assessments of algorithmic bias in large language models (LLMs) are generally catered to uncovering systemic discrimination based on protected characteristics such as sex and ethnicity. However, there are over 180 documented cognitive biases that pervade human reasoning and decision making that are routinely ignored when discussing the ethical complexities of AI. We demonstrate the presence of these cognitive biases in LLMs and discuss the implications of using biased reasoning under the guise of expertise. We call for stronger education, risk management, and continued research as widespread adoption of this technology increases. Finally, we close with a set of best practices for when and how to employ this technology as widespread adoption continues to grow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.13278v1">Integrating LLMs and Decision Transformers for Language Grounded Generative Quality-Diversity</a></div>
    <div class="paper-meta">
      📅 2023-08-25
      | 💬 16 pages, 9 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Quality-Diversity is a branch of stochastic optimization that is often applied to problems from the Reinforcement Learning and control domains in order to construct repertoires of well-performing policies/skills that exhibit diversity with respect to a behavior space. Such archives are usually composed of a finite number of reactive agents which are each associated to a unique behavior descriptor, and instantiating behavior descriptors outside of that coarsely discretized space is not straight-forward. While a few recent works suggest solutions to that issue, the trajectory that is generated is not easily customizable beyond the specification of a target behavior descriptor. We propose to jointly solve those problems in environments where semantic information about static scene elements is available by leveraging a Large Language Model to augment the repertoire with natural language descriptions of trajectories, and training a policy conditioned on those descriptions. Thus, our method allows a user to not only specify an arbitrary target behavior descriptor, but also provide the model with a high-level textual prompt to shape the generated trajectory. We also propose an LLM-based approach to evaluating the performance of such generative agents. Furthermore, we develop a benchmark based on simulated robot navigation in a 2d maze that we use for experimental validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.13062v1">ZeroLeak: Using LLMs for Scalable and Cost Effective Side-Channel Patching</a></div>
    <div class="paper-meta">
      📅 2023-08-24
    </div>
    <details class="paper-abstract">
      Security critical software, e.g., OpenSSL, comes with numerous side-channel leakages left unpatched due to a lack of resources or experts. The situation will only worsen as the pace of code development accelerates, with developers relying on Large Language Models (LLMs) to automatically generate code. In this work, we explore the use of LLMs in generating patches for vulnerable code with microarchitectural side-channel leakages. For this, we investigate the generative abilities of powerful LLMs by carefully crafting prompts following a zero-shot learning approach. All generated code is dynamically analyzed by leakage detection tools, which are capable of pinpointing information leakage at the instruction level leaked either from secret dependent accesses or branches or vulnerable Spectre gadgets, respectively. Carefully crafted prompts are used to generate candidate replacements for vulnerable code, which are then analyzed for correctness and for leakage resilience. From a cost/performance perspective, the GPT4-based configuration costs in API calls a mere few cents per vulnerability fixed. Our results show that LLM-based patching is far more cost-effective and thus provides a scalable solution. Finally, the framework we propose will improve in time, especially as vulnerability detection tools and LLMs mature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.12908v1">POLCA: Power Oversubscription in LLM Cloud Providers</a></div>
    <div class="paper-meta">
      📅 2023-08-24
    </div>
    <details class="paper-abstract">
      Recent innovation in large language models (LLMs), and their myriad use-cases have rapidly driven up the compute capacity demand for datacenter GPUs. Several cloud providers and other enterprises have made substantial plans of growth in their datacenters to support these new workloads. One of the key bottleneck resources in datacenters is power, and given the increasing model sizes of LLMs, they are becoming increasingly power intensive. In this paper, we show that there is a significant opportunity to oversubscribe power in LLM clusters. Power oversubscription improves the power efficiency of these datacenters, allowing more deployable servers per datacenter, and reduces the deployment time, since building new datacenters is slow. We extensively characterize the power consumption patterns of a variety of LLMs and their configurations. We identify the differences between the inference and training power consumption patterns. Based on our analysis of these LLMs, we claim that the average and peak power utilization in LLM clusters for inference should not be very high. Our deductions align with the data from production LLM clusters, revealing that inference workloads offer substantial headroom for power oversubscription. However, the stringent set of telemetry and controls that GPUs offer in a virtualized environment, makes it challenging to have a reliable and robust power oversubscription mechanism. We propose POLCA, our framework for power oversubscription that is robust, reliable, and readily deployable for GPU clusters. Using open-source models to replicate the power patterns observed in production, we simulate POLCA and demonstrate that we can deploy 30% more servers in the same GPU cluster for inference, with minimal performance loss
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.12833v1">Use of LLMs for Illicit Purposes: Threats, Prevention Measures, and Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2023-08-24
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Spurred by the recent rapid increase in the development and distribution of large language models (LLMs) across industry and academia, much recent work has drawn attention to safety- and security-related threats and vulnerabilities of LLMs, including in the context of potentially criminal activities. Specifically, it has been shown that LLMs can be misused for fraud, impersonation, and the generation of malware; while other authors have considered the more general problem of AI alignment. It is important that developers and practitioners alike are aware of security-related problems with such models. In this paper, we provide an overview of existing - predominantly scientific - efforts on identifying and mitigating threats and vulnerabilities arising from LLMs. We present a taxonomy describing the relationship between threats caused by the generative capabilities of LLMs, prevention measures intended to address such threats, and vulnerabilities arising from imperfect prevention measures. With our work, we hope to raise awareness of the limitations of LLMs in light of such security concerns, among both experienced developers and novel users of such technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.12284v1">D4: Improving LLM Pretraining via Document De-Duplication and Diversification</a></div>
    <div class="paper-meta">
      📅 2023-08-23
    </div>
    <details class="paper-abstract">
      Over recent years, an increasing amount of compute and data has been poured into training large language models (LLMs), usually by doing one-pass learning on as many tokens as possible randomly selected from large-scale web corpora. While training on ever-larger portions of the internet leads to consistent performance improvements, the size of these improvements diminishes with scale, and there has been little work exploring the effect of data selection on pre-training and downstream performance beyond simple de-duplication methods such as MinHash. Here, we show that careful data selection (on top of de-duplicated data) via pre-trained model embeddings can speed up training (20% efficiency gains) and improves average downstream accuracy on 16 NLP tasks (up to 2%) at the 6.7B model scale. Furthermore, we show that repeating data intelligently consistently outperforms baseline training (while repeating random data performs worse than baseline training). Our results indicate that clever data selection can significantly improve LLM pre-training, calls into question the common practice of training for a single epoch on as much data as possible, and demonstrates a path to keep improving our models past the limits of randomly sampling web data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.12129v1">Resiliency Analysis of LLM generated models for Industrial Automation</a></div>
    <div class="paper-meta">
      📅 2023-08-23
      | 💬 8 Pages, Conference Manuscript
    </div>
    <details class="paper-abstract">
      This paper proposes a study of the resilience and efficiency of automatically generated industrial automation and control systems using Large Language Models (LLMs). The approach involves modeling the system using percolation theory to estimate its resilience and formulating the design problem as an optimization problem subject to constraints. Techniques from stochastic optimization and regret analysis are used to find a near-optimal solution with provable regret bounds. The study aims to provide insights into the effectiveness and reliability of automatically generated systems in industrial automation and control, and to identify potential areas for improvement in their design and implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.12028v1">LKPNR: LLM and KG for Personalized News Recommendation Framework</a></div>
    <div class="paper-meta">
      📅 2023-08-23
    </div>
    <details class="paper-abstract">
      Accurately recommending candidate news articles to users is a basic challenge faced by personalized news recommendation systems. Traditional methods are usually difficult to grasp the complex semantic information in news texts, resulting in unsatisfactory recommendation results. Besides, these traditional methods are more friendly to active users with rich historical behaviors. However, they can not effectively solve the "long tail problem" of inactive users. To address these issues, this research presents a novel general framework that combines Large Language Models (LLM) and Knowledge Graphs (KG) into semantic representations of traditional methods. In order to improve semantic understanding in complex news texts, we use LLMs' powerful text understanding ability to generate news representations containing rich semantic information. In addition, our method combines the information about news entities and mines high-order structural information through multiple hops in KG, thus alleviating the challenge of long tail distribution. Experimental results demonstrate that compared with various traditional models, the framework significantly improves the recommendation effect. The successful integration of LLM and KG in our framework has established a feasible path for achieving more accurate personalized recommendations in the news field. Our code is available at https://github.com/Xuan-ZW/LKPNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.08239v2">MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-Domain Conversation</a></div>
    <div class="paper-meta">
      📅 2023-08-23
    </div>
    <details class="paper-abstract">
      We propose MemoChat, a pipeline for refining instructions that enables large language models (LLMs) to effectively employ self-composed memos for maintaining consistent long-range open-domain conversations. We demonstrate a long-range open-domain conversation through iterative "memorization-retrieval-response" cycles. This requires us to carefully design tailored tuning instructions for each distinct stage. The instructions are reconstructed from a collection of public datasets to teach the LLMs to memorize and retrieve past dialogues with structured memos, leading to enhanced consistency when participating in future conversations. We invite experts to manually annotate a test set designed to evaluate the consistency of long-range conversations questions. Experiments on three testing scenarios involving both open-source and API-accessible chatbots at scale verify the efficacy of MemoChat, which outperforms strong baselines. Our codes, data and models are available here: https://github.com/LuJunru/MemoChat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.11793v1">Enhancing NeRF akin to Enhancing LLMs: Generalizable NeRF Transformer with Mixture-of-View-Experts</a></div>
    <div class="paper-meta">
      📅 2023-08-22
      | 💬 Accepted by ICCV2023
    </div>
    <details class="paper-abstract">
      Cross-scene generalizable NeRF models, which can directly synthesize novel views of unseen scenes, have become a new spotlight of the NeRF field. Several existing attempts rely on increasingly end-to-end "neuralized" architectures, i.e., replacing scene representation and/or rendering modules with performant neural networks such as transformers, and turning novel view synthesis into a feed-forward inference pipeline. While those feedforward "neuralized" architectures still do not fit diverse scenes well out of the box, we propose to bridge them with the powerful Mixture-of-Experts (MoE) idea from large language models (LLMs), which has demonstrated superior generalization ability by balancing between larger overall model capacity and flexible per-instance specialization. Starting from a recent generalizable NeRF architecture called GNT, we first demonstrate that MoE can be neatly plugged in to enhance the model. We further customize a shared permanent expert and a geometry-aware consistency loss to enforce cross-scene consistency and spatial smoothness respectively, which are essential for generalizable view synthesis. Our proposed model, dubbed GNT with Mixture-of-View-Experts (GNT-MOVE), has experimentally shown state-of-the-art results when transferring to unseen scenes, indicating remarkably better cross-scene generalization in both zero-shot and few-shot settings. Our codes are available at https://github.com/VITA-Group/GNT-MOVE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.10882v1">Giraffe: Adventures in Expanding Context Lengths in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-08-21
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) that rely on attention mechanisms are typically trained with fixed context lengths which enforce upper limits on the length of input sequences that they can handle at evaluation time. To use these models on sequences longer than the train-time context length, one might employ techniques from the growing family of context length extrapolation methods -- most of which focus on modifying the system of positional encodings used in the attention mechanism to indicate where tokens or activations are located in the input sequence. We conduct a wide survey of existing methods of context length extrapolation on a base LLaMA or LLaMA 2 model, and introduce some of our own design as well -- in particular, a new truncation strategy for modifying the basis for the position encoding. We test these methods using three new evaluation tasks (FreeFormQA, AlteredNumericQA, and LongChat-Lines) as well as perplexity, which we find to be less fine-grained as a measure of long context performance of LLMs. We release the three tasks publicly as datasets on HuggingFace. We discover that linear scaling is the best method for extending context length, and show that further gains can be achieved by using longer scales at evaluation time. We also discover promising extrapolation capabilities in the truncated basis. To support further research in this area, we release three new 13B parameter long-context models which we call Giraffe: 4k and 16k context models trained from base LLaMA-13B, and a 32k context model trained from base LLaMA2-13B. We also release the code to replicate our results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.00948v2">Large Linguistic Models: Analyzing theoretical linguistic abilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-08-21
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) has recently improved to the point where the models can perform well on many language tasks. We show here that for the first time, the models can also generate coherent and valid formal analyses of linguistic data and illustrate the vast potential of large language models for analyses of their metalinguistic abilities. LLMs are primarily trained on language data in the form of text; analyzing and evaluating their metalinguistic abilities improves our understanding of their general capabilities and sheds new light on theoretical models in linguistics. In this paper, we probe into GPT-4's metalinguistic capabilities by focusing on three subfields of formal linguistics: syntax, phonology, and semantics. We outline a research program for metalinguistic analyses of large language models, propose experimental designs, provide general guidelines, discuss limitations, and offer future directions for this line of research. This line of inquiry also exemplifies behavioral interpretability of deep learning, where models' representations are accessed by explicit prompting rather than internal representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.10032v1">GameEval: Evaluating LLMs on Conversational Games</a></div>
    <div class="paper-meta">
      📅 2023-08-19
    </div>
    <details class="paper-abstract">
      The rapid advancements in large language models (LLMs) have presented challenges in evaluating those models. Existing evaluation methods are either reference-based or preference based, which inevitably need human intervention or introduce test bias caused by evaluator models. In this paper, we propose GameEval, a novel approach to evaluating LLMs through goal-driven conversational games, overcoming the limitations of previous methods. GameEval treats LLMs as game players and assigns them distinct roles with specific goals achieved by launching conversations of various forms, including discussion, question answering, and voting. We design three unique games with cooperative or adversarial objectives, accompanied by corresponding evaluation metrics, to show how this new paradigm comprehensively evaluates model performance.Through extensive experiments, we show that GameEval can effectively differentiate the capabilities of various LLMs, providing a comprehensive assessment of their integrated abilities to solve complex problems. Our public anonymous code is available at https://github.com/GameEval/GameEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.09954v1">Eva-KELLM: A New Benchmark for Evaluating Knowledge Editing of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-08-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) possess a wealth of knowledge encoded in their parameters. However, this knowledge may become outdated or unsuitable over time. As a result, there has been a growing interest in knowledge editing for LLMs and evaluating its effectiveness. Existing studies primarily focus on knowledge editing using factual triplets, which not only incur high costs for collection but also struggle to express complex facts. Furthermore, these studies are often limited in their evaluation perspectives. In this paper, we propose Eva-KELLM, a new benchmark for evaluating knowledge editing of LLMs. This benchmark includes an evaluation framework and a corresponding dataset. Under our framework, we first ask the LLM to perform knowledge editing using raw documents, which provides a more convenient and universal approach compared to using factual triplets. We then evaluate the updated LLM from multiple perspectives. In addition to assessing the effectiveness of knowledge editing and the retention of unrelated knowledge from conventional studies, we further test the LLM's ability in two aspects: 1) Reasoning with the altered knowledge, aiming for the LLM to genuinely learn the altered knowledge instead of simply memorizing it. 2) Cross-lingual knowledge transfer, where the LLM updated with raw documents in one language should be capable of handling queries from another language. To facilitate further research, we construct and release the corresponding dataset. Using this benchmark, we investigate the effectiveness of several commonly-used knowledge editing methods. Experimental results indicate that the current methods for knowledge editing using raw documents are not effective in yielding satisfactory results, particularly when it comes to reasoning with altered knowledge and cross-lingual knowledge transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.09853v1">How susceptible are LLMs to Logical Fallacies?</a></div>
    <div class="paper-meta">
      📅 2023-08-18
    </div>
    <details class="paper-abstract">
      This paper investigates the rational thinking capability of Large Language Models (LLMs) in multi-round argumentative debates by exploring the impact of fallacious arguments on their logical reasoning performance. More specifically, we present Logic Competence Measurement Benchmark (LOGICOM), a diagnostic benchmark to assess the robustness of LLMs against logical fallacies. LOGICOM involves two agents: a persuader and a debater engaging in a multi-round debate on a controversial topic, where the persuader tries to convince the debater of the correctness of its claim. First, LOGICOM assesses the potential of LLMs to change their opinions through reasoning. Then, it evaluates the debater's performance in logical reasoning by contrasting the scenario where the persuader employs logical fallacies against one where logical reasoning is used. We use this benchmark to evaluate the performance of GPT-3.5 and GPT-4 using a dataset containing controversial topics, claims, and reasons supporting them. Our findings indicate that both GPT-3.5 and GPT-4 can adjust their opinion through reasoning. However, when presented with logical fallacies, GPT-3.5 and GPT-4 are erroneously convinced 41% and 69% more often, respectively, compared to when logical reasoning is used. Finally, we introduce a new dataset containing over 5k pairs of logical vs. fallacious arguments. The source code and dataset of this work are made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.03333v2">Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM</a></div>
    <div class="paper-meta">
      📅 2023-08-18
      | 💬 Accepted at RecSys 2023
    </div>
    <details class="paper-abstract">
      The analysis and mining of user heterogeneous behavior are of paramount importance in recommendation systems. However, the conventional approach of incorporating various types of heterogeneous behavior into recommendation models leads to feature sparsity and knowledge fragmentation issues. To address this challenge, we propose a novel approach for personalized recommendation via Large Language Model (LLM), by extracting and fusing heterogeneous knowledge from user heterogeneous behavior information. In addition, by combining heterogeneous knowledge and recommendation tasks, instruction tuning is performed on LLM for personalized recommendations. The experimental results demonstrate that our method can effectively integrate user heterogeneous behavior and significantly improve recommendation performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.13923v2">GrammarGPT: Exploring Open-Source LLMs for Native Chinese Grammatical Error Correction with Supervised Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2023-08-17
    </div>
    <details class="paper-abstract">
      Grammatical error correction aims to correct ungrammatical sentences automatically. Recently, some work has demonstrated the excellent capabilities of closed-source Large Language Models (LLMs, e.g., ChatGPT) in grammatical error correction. However, the potential of open-source LLMs remains unexplored. In this paper, we introduced GrammarGPT, an open-source LLM, to preliminary explore its potential for native Chinese grammatical error correction. The core recipe of GrammarGPT is to leverage the hybrid dataset of ChatGPT-generated and human-annotated. For grammatical errors with clues, we proposed a heuristic method to guide ChatGPT to generate ungrammatical sentences by providing those clues. For grammatical errors without clues, we collected ungrammatical sentences from publicly available websites and manually corrected them. In addition, we employed an error-invariant augmentation method to enhance the ability of the model to correct native Chinese grammatical errors. We ultimately constructed about 1k parallel data and utilized these data to fine-tune open-source LLMs (e.g., Phoenix, released by The Chinese University of Hong Kong, Shenzhen) with instruction tuning. The experimental results show that GrammarGPT outperforms the existing SOTA system significantly. Although model parameters are 20x larger than the SOTA baseline, the required amount of data for instruction tuning is 1200x smaller, illustrating the potential of open-source LLMs on native CGEC. Our GrammarGPT ranks $3^{rd}$ on NLPCC2023 SharedTask1, demonstrating our approach's effectiveness. The code and data are available at \url{https://github.com/FreedomIntelligence/GrammarGPT}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.11584v1">Building Emotional Support Chatbots in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-08-17
    </div>
    <details class="paper-abstract">
      The integration of emotional support into various conversational scenarios presents profound societal benefits, such as social interactions, mental health counseling, and customer service. However, there are unsolved challenges that hinder real-world applications in this field, including limited data availability and the absence of well-accepted model training paradigms. This work endeavors to navigate these challenges by harnessing the capabilities of Large Language Models (LLMs). We introduce an innovative methodology that synthesizes human insights with the computational prowess of LLMs to curate an extensive emotional support dialogue dataset. Our approach is initiated with a meticulously designed set of dialogues spanning diverse scenarios as generative seeds. By utilizing the in-context learning potential of ChatGPT, we recursively generate an ExTensible Emotional Support dialogue dataset, named ExTES. Following this, we deploy advanced tuning techniques on the LLaMA model, examining the impact of diverse training strategies, ultimately yielding an LLM meticulously optimized for emotional support interactions. An exhaustive assessment of the resultant model showcases its proficiency in offering emotional support, marking a pivotal step in the realm of emotional support bots and paving the way for subsequent research and implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.08728v1">LLM-FuncMapper: Function Identification for Interpreting Complex Clauses in Building Codes via LLM</a></div>
    <div class="paper-meta">
      📅 2023-08-17
    </div>
    <details class="paper-abstract">
      As a vital stage of automated rule checking (ARC), rule interpretation of regulatory texts requires considerable effort. However, interpreting regulatory clauses with implicit properties or complex computational logic is still challenging due to the lack of domain knowledge and limited expressibility of conventional logic representations. Thus, LLM-FuncMapper, an approach to identifying predefined functions needed to interpret various regulatory clauses based on the large language model (LLM), is proposed. First, by systematically analysis of building codes, a series of atomic functions are defined to capture shared computational logics of implicit properties and complex constraints, creating a database of common blocks for interpreting regulatory clauses. Then, a prompt template with the chain of thought is developed and further enhanced with a classification-based tuning strategy, to enable common LLMs for effective function identification. Finally, the proposed approach is validated with statistical analysis, experiments, and proof of concept. Statistical analysis reveals a long-tail distribution and high expressibility of the developed function database, with which almost 100% of computer-processible clauses can be interpreted and represented as computer-executable codes. Experiments show that LLM-FuncMapper achieve promising results in identifying relevant predefined functions for rule interpretation. Further proof of concept in automated rule interpretation also demonstrates the possibility of LLM-FuncMapper in interpreting complex regulatory clauses. To the best of our knowledge, this study is the first attempt to introduce LLM for understanding and interpreting complex regulatory clauses, which may shed light on further adoption of LLM in the construction domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.09723v1">FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2023-08-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved state-of-the-art performance across various language tasks but pose challenges for practical deployment due to their substantial memory requirements. Furthermore, the latest generative models suffer from high inference costs caused by the memory bandwidth bottleneck in the auto-regressive decoding process. To address these issues, we propose an efficient weight-only quantization method that reduces memory consumption and accelerates inference for LLMs. To ensure minimal quality degradation, we introduce a simple and effective heuristic approach that utilizes only the model weights of a pre-trained model. This approach is applicable to both Mixture-of-Experts (MoE) and dense models without requiring additional fine-tuning. To demonstrate the effectiveness of our proposed method, we first analyze the challenges and issues associated with LLM quantization. Subsequently, we present our heuristic approach, which adaptively finds the granularity of quantization, effectively addressing these problems. Furthermore, we implement highly efficient GPU GEMMs that perform on-the-fly matrix multiplication and dequantization, supporting the multiplication of fp16 or bf16 activations with int8 or int4 weights. We evaluate our approach on large-scale open source models such as OPT-175B and internal MoE models, showcasing minimal accuracy loss while achieving up to 3.65 times higher throughput on the same number of GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.00108v2">Better patching using LLM prompting, via Self-Consistency</a></div>
    <div class="paper-meta">
      📅 2023-08-16
      | 💬 Accepted at ASE-NIER (2023) track
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) can be induced to solve non-trivial problems with "few-shot" prompts including illustrative problem-solution examples. Now if the few-shots also include "chain of thought" (CoT) explanations, which are of the form problem-explanation-solution, LLMs will generate a "explained" solution, and perform even better. Recently an exciting, substantially better technique, self-consistency [1] (S-C) has emerged, based on the intuition that there are many plausible explanations for the right solution; when the LLM is sampled repeatedly to generate a pool of explanation-solution pairs, for a given problem, the most frequently occurring solutions in the pool (ignoring the explanations) tend to be even more likely to be correct! Unfortunately, the use of this highly-performant S-C (or even CoT) approach in software engineering settings is hampered by the lack of explanations; most software datasets lack explanations. In this paper, we describe an application of the S-C approach to program repair, using the commit log on the fix as the explanation, only in the illustrative few-shots. We achieve state-of-the art results, beating previous approaches to prompting-based program repair, on the MODIT dataset; we also find evidence suggesting that the correct commit messages are helping the LLM learn to produce better patches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.11787v2">LLM Cognitive Judgements Differ From Human</a></div>
    <div class="paper-meta">
      📅 2023-08-16
      | 💬 7 pages, 1 figure. License changed to CC BY-NC-SA
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have lately been on the spotlight of researchers, businesses, and consumers alike. While the linguistic capabilities of such models have been studied extensively, there is growing interest in investigating them as cognitive subjects. In the present work I examine GPT-3 and ChatGPT capabilities on an limited-data inductive reasoning task from the cognitive science literature. The results suggest that these models' cognitive judgements are not human-like.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.07968v1">Teach LLMs to Personalize -- An Approach inspired by Writing Education</a></div>
    <div class="paper-meta">
      📅 2023-08-15
    </div>
    <details class="paper-abstract">
      Personalized text generation is an emerging research area that has attracted much attention in recent years. Most studies in this direction focus on a particular domain by designing bespoke features or models. In this work, we propose a general approach for personalized text generation using large language models (LLMs). Inspired by the practice of writing education, we develop a multistage and multitask framework to teach LLMs for personalized generation. In writing instruction, the task of writing from sources is often decomposed into multiple steps that involve finding, evaluating, summarizing, synthesizing, and integrating information. Analogously, our approach to personalized text generation consists of multiple stages: retrieval, ranking, summarization, synthesis, and generation. In addition, we introduce a multitask setting that helps the model improve its generation ability further, which is inspired by the observation in education that a student's reading proficiency and writing ability are often correlated. We evaluate our approach on three public datasets, each of which covers a different and representative domain. Our results show significant improvements over a variety of baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.07891v1">Link-Context Learning for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2023-08-15
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The ability to learn from context with novel concepts, and deliver appropriate responses are essential in human conversations. Despite current Multimodal Large Language Models (MLLMs) and Large Language Models (LLMs) being trained on mega-scale datasets, recognizing unseen images or understanding novel concepts in a training-free manner remains a challenge. In-Context Learning (ICL) explores training-free few-shot learning, where models are encouraged to ``learn to learn" from limited tasks and generalize to unseen tasks. In this work, we propose link-context learning (LCL), which emphasizes "reasoning from cause and effect" to augment the learning capabilities of MLLMs. LCL goes beyond traditional ICL by explicitly strengthening the causal relationship between the support set and the query set. By providing demonstrations with causal links, LCL guides the model to discern not only the analogy but also the underlying causal associations between data points, which empowers MLLMs to recognize unseen images and understand novel concepts more effectively. To facilitate the evaluation of this novel approach, we introduce the ISEKAI dataset, comprising exclusively of unseen generated image-label pairs designed for link-context learning. Extensive experiments show that our LCL-MLLM exhibits strong link-context learning capabilities to novel concepts over vanilla MLLMs. Code and data will be released at https://github.com/isekai-portal/Link-Context-Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.07635v1">LLM-Mini-CEX: Automatic Evaluation of Large Language Model for Diagnostic Conversation</a></div>
    <div class="paper-meta">
      📅 2023-08-15
    </div>
    <details class="paper-abstract">
      There is an increasing interest in developing LLMs for medical diagnosis to improve diagnosis efficiency. Despite their alluring technological potential, there is no unified and comprehensive evaluation criterion, leading to the inability to evaluate the quality and potential risks of medical LLMs, further hindering the application of LLMs in medical treatment scenarios. Besides, current evaluations heavily rely on labor-intensive interactions with LLMs to obtain diagnostic dialogues and human evaluation on the quality of diagnosis dialogue. To tackle the lack of unified and comprehensive evaluation criterion, we first initially establish an evaluation criterion, termed LLM-specific Mini-CEX to assess the diagnostic capabilities of LLMs effectively, based on original Mini-CEX. To address the labor-intensive interaction problem, we develop a patient simulator to engage in automatic conversations with LLMs, and utilize ChatGPT for evaluating diagnosis dialogues automatically. Experimental results show that the LLM-specific Mini-CEX is adequate and necessary to evaluate medical diagnosis dialogue. Besides, ChatGPT can replace manual evaluation on the metrics of humanistic qualities and provides reproducible and automated comparisons between different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.07540v1">CALYPSO: LLMs as Dungeon Masters' Assistants</a></div>
    <div class="paper-meta">
      📅 2023-08-15
      | 💬 11 pages, 4 figures. AIIDE 2023
    </div>
    <details class="paper-abstract">
      The role of a Dungeon Master, or DM, in the game Dungeons & Dragons is to perform multiple tasks simultaneously. The DM must digest information about the game setting and monsters, synthesize scenes to present to other players, and respond to the players' interactions with the scene. Doing all of these tasks while maintaining consistency within the narrative and story world is no small feat of human cognition, making the task tiring and unapproachable to new players. Large language models (LLMs) like GPT-3 and ChatGPT have shown remarkable abilities to generate coherent natural language text. In this paper, we conduct a formative evaluation with DMs to establish the use cases of LLMs in D&D and tabletop gaming generally. We introduce CALYPSO, a system of LLM-powered interfaces that support DMs with information and inspiration specific to their own scenario. CALYPSO distills game context into bite-sized prose and helps brainstorm ideas without distracting the DM from the game. When given access to CALYPSO, DMs reported that it generated high-fidelity text suitable for direct presentation to players, and low-fidelity ideas that the DM could develop further while maintaining their creative agency. We see CALYPSO as exemplifying a paradigm of AI-augmented tools that provide synchronous creative assistance within established game worlds, and tabletop gaming more broadly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2308.06391v1">Dynamic Planning with a LLM</a></div>
    <div class="paper-meta">
      📅 2023-08-15
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) can solve many NLP tasks in zero-shot settings, applications involving embodied agents remain problematic. In particular, complex plans that require multi-step reasoning become difficult and too costly as the context window grows. Planning requires understanding the likely effects of one's actions and identifying whether the current environment satisfies the goal state. While symbolic planners find optimal solutions quickly, they require a complete and accurate representation of the planning problem, severely limiting their use in practical scenarios. In contrast, modern LLMs cope with noisy observations and high levels of uncertainty when reasoning about a task. Our work presents LLM Dynamic Planner (LLM-DP): a neuro-symbolic framework where an LLM works hand-in-hand with a traditional planner to solve an embodied task. Given action-descriptions, LLM-DP solves Alfworld faster and more efficiently than a naive LLM ReAct baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.07201v1">ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate</a></div>
    <div class="paper-meta">
      📅 2023-08-14
    </div>
    <details class="paper-abstract">
      Text evaluation has historically posed significant challenges, often demanding substantial labor and time cost. With the emergence of large language models (LLMs), researchers have explored LLMs' potential as alternatives for human evaluation. While these single-agent-based approaches show promise, experimental results suggest that further advancements are needed to bridge the gap between their current effectiveness and human-level evaluation quality. Recognizing that best practices of human evaluation processes often involve multiple human annotators collaborating in the evaluation, we resort to a multi-agent debate framework, moving beyond single-agent prompting strategies. The multi-agent-based approach enables a group of LLMs to synergize with an array of intelligent counterparts, harnessing their distinct capabilities and expertise to enhance efficiency and effectiveness in handling intricate tasks. In this paper, we construct a multi-agent referee team called ChatEval to autonomously discuss and evaluate the quality of generated responses from different models on open-ended questions and traditional natural language generation (NLG) tasks. Our analysis shows that ChatEval transcends mere textual scoring, offering a human-mimicking evaluation process for reliable assessments. Our code is available at https://github.com/chanchimin/ChatEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.01861v2">ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation</a></div>
    <div class="paper-meta">
      📅 2023-08-14
    </div>
    <details class="paper-abstract">
      In this work, we make the first attempt to evaluate LLMs in a more challenging code generation scenario, i.e. class-level code generation. We first manually construct the first class-level code generation benchmark ClassEval of 100 class-level Python code generation tasks with approximately 500 person-hours. Based on it, we then perform the first study of 11 state-of-the-art LLMs on class-level code generation. Based on our results, we have the following main findings. First, we find that all existing LLMs show much worse performance on class-level code generation compared to on standalone method-level code generation benchmarks like HumanEval; and the method-level coding ability cannot equivalently reflect the class-level coding ability among LLMs. Second, we find that GPT-4 and GPT-3.5 still exhibit dominate superior than other LLMs on class-level code generation, and the second-tier models includes Instruct-Starcoder, Instruct-Codegen, and Wizardcoder with very similar performance. Third, we find that generating the entire class all at once (i.e. holistic generation strategy) is the best generation strategy only for GPT-4 and GPT-3.5, while method-by-method generation (i.e. incremental and compositional) is better strategies for the other models with limited ability of understanding long instructions and utilizing the middle information. Lastly, we find the limited model ability of generating method-dependent code and discuss the frequent error types in generated classes. Our benchmark is available at https://github.com/FudanSELab/ClassEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.06932v1">DIVAS: An LLM-based End-to-End Framework for SoC Security Analysis and Policy-based Protection</a></div>
    <div class="paper-meta">
      📅 2023-08-14
      | 💬 15 pages, 7 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Securing critical assets in a bus-based System-On-Chip (SoC) is imperative to mitigate potential vulnerabilities and prevent unauthorized access, ensuring the integrity, availability, and confidentiality of the system. Ensuring security throughout the SoC design process is a formidable task owing to the inherent intricacies in SoC designs and the dispersion of assets across diverse IPs. Large Language Models (LLMs), exemplified by ChatGPT (OpenAI) and BARD (Google), have showcased remarkable proficiency across various domains, including security vulnerability detection and prevention in SoC designs. In this work, we propose DIVAS, a novel framework that leverages the knowledge base of LLMs to identify security vulnerabilities from user-defined SoC specifications, map them to the relevant Common Weakness Enumerations (CWEs), followed by the generation of equivalent assertions, and employ security measures through enforcement of security policies. The proposed framework is implemented using multiple ChatGPT and BARD models, and their performance was analyzed while generating relevant CWEs from the SoC specifications provided. The experimental results obtained from open-source SoC benchmarks demonstrate the efficacy of our proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.13534v1">Building Trust in Conversational AI: A Comprehensive Review and Solution Architecture for Explainable, Privacy-Aware Systems using LLMs and Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2023-08-13
    </div>
    <details class="paper-abstract">
      Conversational AI systems have emerged as key enablers of human-like interactions across diverse sectors. Nevertheless, the balance between linguistic nuance and factual accuracy has proven elusive. In this paper, we first introduce LLMXplorer, a comprehensive tool that provides an in-depth review of over 150 Large Language Models (LLMs), elucidating their myriad implications ranging from social and ethical to regulatory, as well as their applicability across industries. Building on this foundation, we propose a novel functional architecture that seamlessly integrates the structured dynamics of Knowledge Graphs with the linguistic capabilities of LLMs. Validated using real-world AI news data, our architecture adeptly blends linguistic sophistication with factual rigour and further strengthens data security through Role-Based Access Control. This research provides insights into the evolving landscape of conversational AI, emphasizing the imperative for systems that are efficient, transparent, and trustworthy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.03987v2">A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation</a></div>
    <div class="paper-meta">
      📅 2023-08-12
      | 💬 update to include additional experiments
    </div>
    <details class="paper-abstract">
      Recently developed large language models have achieved remarkable success in generating fluent and coherent text. However, these models often tend to 'hallucinate' which critically hampers their reliability. In this work, we address this crucial problem and propose an approach that actively detects and mitigates hallucinations during the generation process. Specifically, we first identify the candidates of potential hallucination leveraging the model's logit output values, check their correctness through a validation procedure, mitigate the detected hallucinations, and then continue with the generation process. Through extensive experiments with GPT-3.5 (text-davinci-003) on the 'article generation task', we first demonstrate the individual efficacy of our detection and mitigation techniques. Specifically, the detection technique achieves a recall of ~88% and the mitigation technique successfully mitigates 57.6% of the correctly detected hallucinations. Importantly, our mitigation technique does not introduce new hallucinations even in the case of incorrectly detected hallucinations, i.e., false positives. Then, we show that the proposed active detection and mitigation approach successfully reduces the hallucinations of the GPT-3.5 model from 47.5% to 14.5% on average. We further demonstrate the effectiveness and wide applicability of our approach through additional studies including performance on different types of questions (multi-hop and false premise questions) and with another LLM from a different model family (Vicuna). In summary, our work contributes to improving the reliability and trustworthiness of large language models, a crucial step en route to enabling their widespread adoption in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.06391v1">Dynamic Planning with a LLM</a></div>
    <div class="paper-meta">
      📅 2023-08-11
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) can solve many NLP tasks in zero-shot settings, applications involving embodied agents remain problematic. In particular, complex plans that require multi-step reasoning become difficult and too costly as the context window grows. Planning requires understanding the likely effects of one's actions and identifying whether the current environment satisfies the goal state. While symbolic planners find optimal solutions quickly, they require a complete and accurate representation of the planning problem, severely limiting their use in practical scenarios. In contrast, modern LLMs cope with noisy observations and high levels of uncertainty when reasoning about a task. Our work presents LLM Dynamic Planner (LLM-DP): a neuro-symbolic framework where an LLM works hand-in-hand with a traditional planner to solve an embodied task. Given action-descriptions, LLM-DP solves Alfworld faster and more efficiently than a naive LLM ReAct baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.05481v2">LLM As DBA</a></div>
    <div class="paper-meta">
      📅 2023-08-11
    </div>
    <details class="paper-abstract">
      Database administrators (DBAs) play a crucial role in managing, maintaining and optimizing a database system to ensure data availability, performance, and reliability. However, it is hard and tedious for DBAs to manage a large number of database instances (e.g., millions of instances on the cloud databases). Recently large language models (LLMs) have shown great potential to understand valuable documents and accordingly generate reasonable answers. Thus, we propose D-Bot, a LLM-based database administrator that can continuously acquire database maintenance experience from textual sources, and provide reasonable, well-founded, in-time diagnosis and optimization advice for target databases. This paper presents a revolutionary LLM-centric framework for database maintenance, including (i) database maintenance knowledge detection from documents and tools, (ii) tree of thought reasoning for root cause analysis, and (iii) collaborative diagnosis among multiple LLMs. Our preliminary experimental results that D-Bot can efficiently and effectively diagnose the root causes and our code is available at github.com/TsinghuaDatabaseGroup/DB-GPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.05960v1">BOLAA: Benchmarking and Orchestrating LLM-augmented Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2023-08-11
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The massive successes of large language models (LLMs) encourage the emerging exploration of LLM-augmented Autonomous Agents (LAAs). An LAA is able to generate actions with its core LLM and interact with environments, which facilitates the ability to resolve complex tasks by conditioning on past interactions such as observations and actions. Since the investigation of LAA is still very recent, limited explorations are available. Therefore, we provide a comprehensive comparison of LAA in terms of both agent architectures and LLM backbones. Additionally, we propose a new strategy to orchestrate multiple LAAs such that each labor LAA focuses on one type of action, \textit{i.e.} BOLAA, where a controller manages the communication among multiple agents. We conduct simulations on both decision-making and multi-step reasoning environments, which comprehensively justify the capacity of LAAs. Our performance results provide quantitative suggestions for designing LAA architectures and the optimal choice of LLMs, as well as the compatibility of both. We release our implementation code of LAAs to the public at \url{https://github.com/salesforce/BOLAA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.05391v1">Enhancing Trust in LLM-Based AI Automation Agents: New Considerations and Future Challenges</a></div>
    <div class="paper-meta">
      📅 2023-08-10
      | 💬 Accepted to the First International Workshop on the Future of No-Code Digital Apprentices
    </div>
    <details class="paper-abstract">
      Trust in AI agents has been extensively studied in the literature, resulting in significant advancements in our understanding of this field. However, the rapid advancements in Large Language Models (LLMs) and the emergence of LLM-based AI agent frameworks pose new challenges and opportunities for further research. In the field of process automation, a new generation of AI-based agents has emerged, enabling the execution of complex tasks. At the same time, the process of building automation has become more accessible to business users via user-friendly no-code tools and training mechanisms. This paper explores these new challenges and opportunities, analyzes the main aspects of trust in AI agents discussed in existing literature, and identifies specific considerations and challenges relevant to this new generation of automation agents. We also evaluate how nascent products in this category address these considerations. Finally, we highlight several challenges that the research community should address in this evolving landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.05177v1">Fixing Rust Compilation Errors using LLMs</a></div>
    <div class="paper-meta">
      📅 2023-08-09
    </div>
    <details class="paper-abstract">
      The Rust programming language, with its safety guarantees, has established itself as a viable choice for low-level systems programming language over the traditional, unsafe alternatives like C/C++. These guarantees come from a strong ownership-based type system, as well as primitive support for features like closures, pattern matching, etc., that make the code more concise and amenable to reasoning. These unique Rust features also pose a steep learning curve for programmers. This paper presents a tool called RustAssistant that leverages the emergent capabilities of Large Language Models (LLMs) to automatically suggest fixes for Rust compilation errors. RustAssistant uses a careful combination of prompting techniques as well as iteration with an LLM to deliver high accuracy of fixes. RustAssistant is able to achieve an impressive peak accuracy of roughly 74% on real-world compilation errors in popular open-source Rust repositories. We plan to release our dataset of Rust compilation errors to enable further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.04624v1">Benchmarking LLM powered Chatbots: Methods and Metrics</a></div>
    <div class="paper-meta">
      📅 2023-08-08
      | 💬 8 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Autonomous conversational agents, i.e. chatbots, are becoming an increasingly common mechanism for enterprises to provide support to customers and partners. In order to rate chatbots, especially ones powered by Generative AI tools like Large Language Models (LLMs) we need to be able to accurately assess their performance. This is where chatbot benchmarking becomes important. In this paper, we propose the use of a novel benchmark that we call the E2E (End to End) benchmark, and show how the E2E benchmark can be used to evaluate accuracy and usefulness of the answers provided by chatbots, especially ones powered by LLMs. We evaluate an example chatbot at different levels of sophistication based on both our E2E benchmark, as well as other available metrics commonly used in the state of art, and observe that the proposed benchmark show better results compared to others. In addition, while some metrics proved to be unpredictable, the metric associated with the E2E benchmark, which uses cosine similarity performed well in evaluating chatbots. The performance of our best models shows that there are several benefits of using the cosine similarity score as a metric in the E2E benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.04623v1">Accelerating LLM Inference with Staged Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2023-08-08
      | 💬 Published at ES-FOMO at ICML 2023
    </div>
    <details class="paper-abstract">
      Recent advances with large language models (LLM) illustrate their diverse capabilities. We propose a novel algorithm, staged speculative decoding, to accelerate LLM inference in small-batch, on-device scenarios. We address the low arithmetic intensity of small-batch inference by improving upon previous work in speculative decoding. First, we restructure the speculative batch as a tree, which reduces generation costs and increases the expected tokens per batch. Second, we add a second stage of speculative decoding. Taken together, we reduce single-batch decoding latency by 3.16x with a 762M parameter GPT-2-L model while perfectly preserving output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.01941v2">AI Transparency in the Age of LLMs: A Human-Centered Research Roadmap</a></div>
    <div class="paper-meta">
      📅 2023-08-08
    </div>
    <details class="paper-abstract">
      The rise of powerful large language models (LLMs) brings about tremendous opportunities for innovation but also looming risks for individuals and society at large. We have reached a pivotal moment for ensuring that LLMs and LLM-infused applications are developed and deployed responsibly. However, a central pillar of responsible AI -- transparency -- is largely missing from the current discourse around LLMs. It is paramount to pursue new approaches to provide transparency for LLMs, and years of research at the intersection of AI and human-computer interaction (HCI) highlight that we must do so with a human-centered perspective: Transparency is fundamentally about supporting appropriate human understanding, and this understanding is sought by different stakeholders with different goals in different contexts. In this new era of LLMs, we must develop and design approaches to transparency by considering the needs of stakeholders in the emerging LLM ecosystem, the novel types of LLM-infused applications being built, and the new usage patterns and challenges around LLMs, all while building on lessons learned about how people process, interact with, and make use of information. We reflect on the unique challenges that arise in providing transparency for LLMs, along with lessons learned from HCI and responsible AI research that has taken a human-centered perspective on AI transparency. We then lay out four common approaches that the community has taken to achieve transparency -- model reporting, publishing evaluation results, providing explanations, and communicating uncertainty -- and call out open questions around how these approaches may or may not be applied to LLMs. We hope this provides a starting point for discussion and a useful roadmap for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.01157v2">LLMs Understand Glass-Box Models, Discover Surprises, and Suggest Repairs</a></div>
    <div class="paper-meta">
      📅 2023-08-07
    </div>
    <details class="paper-abstract">
      We show that large language models (LLMs) are remarkably good at working with interpretable models that decompose complex outcomes into univariate graph-represented components. By adopting a hierarchical approach to reasoning, LLMs can provide comprehensive model-level summaries without ever requiring the entire model to fit in context. This approach enables LLMs to apply their extensive background knowledge to automate common tasks in data science such as detecting anomalies that contradict prior knowledge, describing potential reasons for the anomalies, and suggesting repairs that would remove the anomalies. We use multiple examples in healthcare to demonstrate the utility of these new capabilities of LLMs, with particular emphasis on Generalized Additive Models (GAMs). Finally, we present the package $\texttt{TalkToEBM}$ as an open-source LLM-GAM interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.03107v1">Embedding-based Retrieval with LLM for Effective Agriculture Information Extracting from Unstructured Data</a></div>
    <div class="paper-meta">
      📅 2023-08-06
    </div>
    <details class="paper-abstract">
      Pest identification is a crucial aspect of pest control in agriculture. However, most farmers are not capable of accurately identifying pests in the field, and there is a limited number of structured data sources available for rapid querying. In this work, we explored using domain-agnostic general pre-trained large language model(LLM) to extract structured data from agricultural documents with minimal or no human intervention. We propose a methodology that involves text retrieval and filtering using embedding-based retrieval, followed by LLM question-answering to automatically extract entities and attributes from the documents, and transform them into structured data. In comparison to existing methods, our approach achieves consistently better accuracy in the benchmark while maintaining efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.04416v1">Legal Summarisation through LLMs: The PRODIGIT Project</a></div>
    <div class="paper-meta">
      📅 2023-08-04
      | 💬 arXiv admin note: substantial text overlap with arXiv:2303.09136 by other authors
    </div>
    <details class="paper-abstract">
      We present some initial results of a large-scale Italian project called PRODIGIT which aims to support tax judges and lawyers through digital technology, focusing on AI. We have focused on generation of summaries of judicial decisions and on the extraction of related information, such as the identification of legal issues and decision-making criteria, and the specification of keywords. To this end, we have deployed and evaluated different tools and approaches to extractive and abstractive summarisation. We have applied LLMs, and particularly on GPT4, which has enabled us to obtain results that proved satisfactory, according to an evaluation by expert tax judges and lawyers. On this basis, a prototype application is being built which will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2308.01862v1">Wider and Deeper LLM Networks are Fairer LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2023-08-04
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Measuring the quality of responses generated by LLMs is a challenging task, particularly when it comes to evaluating whether the response is aligned with human preference. A novel approach involves using the LLM itself to make evaluation and stabilizing the results through multiple independent evaluations, similar to a single-layer narrow LLM network. This network consists of a fixed number of neurons, with each neuron being the same LLM. In this paper, we draw upon the extensive research on deep neural networks to explore whether deeper and wider networks can lead to fairer evaluations. Specifically, inspired by the observation that different neurons in a neural network are responsible for detecting different concepts, we first adaptively generate as many neuron roles as possible for each evaluation sample. Each perspective corresponds to the role of a specific LLM neuron in the first layer. In subsequent layers, we follow the idea that higher layers in deep networks are responsible for more comprehensive features, each layer receives representations from all neurons in the previous layer, integrating the locally learned evaluation information to obtain a more comprehensive evaluation result. Interestingly, this network design resembles the process of academic paper reviewing. To validate the effectiveness of our method, we construct the largest and most diverse English evaluation benchmark LLMEval$^2$ for LLM evaluators, comprising 15 tasks, 8 abilities, and 2,553 samples. Experimental results demonstrate that a wider network (involving many reviewers) with 2 layers (one round of discussion) performs the best, improving kappa correlation coefficient from 0.28 to 0.34. We also leverage WideDeep to aid in the assessment of Chinese LLMs, which has accelerated the evaluation time by 4.6 times, resulting in a 60% cost saving. WideDeep achieves a remarkable 93% agreement level among humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.01862v1">Wider and Deeper LLM Networks are Fairer LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2023-08-03
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Measuring the quality of responses generated by LLMs is a challenging task, particularly when it comes to evaluating whether the response is aligned with human preference. A novel approach involves using the LLM itself to make evaluation and stabilizing the results through multiple independent evaluations, similar to a single-layer narrow LLM network. This network consists of a fixed number of neurons, with each neuron being the same LLM. In this paper, we draw upon the extensive research on deep neural networks to explore whether deeper and wider networks can lead to fairer evaluations. Specifically, inspired by the observation that different neurons in a neural network are responsible for detecting different concepts, we first adaptively generate as many neuron roles as possible for each evaluation sample. Each perspective corresponds to the role of a specific LLM neuron in the first layer. In subsequent layers, we follow the idea that higher layers in deep networks are responsible for more comprehensive features, each layer receives representations from all neurons in the previous layer, integrating the locally learned evaluation information to obtain a more comprehensive evaluation result. Interestingly, this network design resembles the process of academic paper reviewing. To validate the effectiveness of our method, we construct the largest and most diverse English evaluation benchmark LLMEval$^2$ for LLM evaluators, comprising 15 tasks, 8 abilities, and 2,553 samples. Experimental results demonstrate that a wider network (involving many reviewers) with 2 layers (one round of discussion) performs the best, improving kappa correlation coefficient from 0.28 to 0.34. We also leverage WideDeep to aid in the assessment of Chinese LLMs, which has accelerated the evaluation time by 4.6 times, resulting in a 60% cost saving. WideDeep achieves a remarkable 93% agreement level among humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.06556v2">Are LLMs All You Need for Task-Oriented Dialogue?</a></div>
    <div class="paper-meta">
      📅 2023-08-03
      | 💬 Accepted to SIGDial 2023
    </div>
    <details class="paper-abstract">
      Instructions-tuned Large Language Models (LLMs) gained recently huge popularity thanks to their ability to interact with users through conversation. In this work we aim to evaluate their ability to complete multi-turn tasks and interact with external databases in the context of established task-oriented dialogue benchmarks. We show that for explicit belief state tracking, LLMs underperform compared to specialized task-specific models. Nevertheless, they show ability to guide the dialogue to successful ending if given correct slot values. Furthermore this ability improves with access to true belief state distribution or in-domain examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2308.01862v1">Wider and Deeper LLM Networks are Fairer LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2023-08-03
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Measuring the quality of responses generated by LLMs is a challenging task, particularly when it comes to evaluating whether the response is aligned with human preference. A novel approach involves using the LLM itself to make evaluation and stabilizing the results through multiple independent evaluations, similar to a single-layer narrow LLM network. This network consists of a fixed number of neurons, with each neuron being the same LLM. In this paper, we draw upon the extensive research on deep neural networks to explore whether deeper and wider networks can lead to fairer evaluations. Specifically, inspired by the observation that different neurons in a neural network are responsible for detecting different concepts, we first adaptively generate as many neuron roles as possible for each evaluation sample. Each perspective corresponds to the role of a specific LLM neuron in the first layer. In subsequent layers, we follow the idea that higher layers in deep networks are responsible for more comprehensive features, each layer receives representations from all neurons in the previous layer, integrating the locally learned evaluation information to obtain a more comprehensive evaluation result. Interestingly, this network design resembles the process of academic paper reviewing. To validate the effectiveness of our method, we construct the largest and most diverse English evaluation benchmark LLMEval$^2$ for LLM evaluators, comprising 15 tasks, 8 abilities, and 2,553 samples. Experimental results demonstrate that a wider network (involving many reviewers) with 2 layers (one round of discussion) performs the best, improving kappa correlation coefficient from 0.28 to 0.34. We also leverage WideDeep to aid in the assessment of Chinese LLMs, which has accelerated the evaluation time by 4.6 times, resulting in a 60% cost saving. WideDeep achieves a remarkable 93% agreement level among humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.16125v2">SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension</a></div>
    <div class="paper-meta">
      📅 2023-08-02
      | 💬 Technical Report; Project released at: https://github.com/AILab-CVC/SEED-Bench
    </div>
    <details class="paper-abstract">
      Based on powerful Large Language Models (LLMs), recent generative Multimodal Large Language Models (MLLMs) have gained prominence as a pivotal research area, exhibiting remarkable capability for both comprehension and generation. In this work, we address the evaluation of generative comprehension in MLLMs as a preliminary step towards a comprehensive assessment of generative models, by introducing a benchmark named SEED-Bench. SEED-Bench consists of 19K multiple choice questions with accurate human annotations (x 6 larger than existing benchmarks), which spans 12 evaluation dimensions including the comprehension of both the image and video modality. We develop an advanced pipeline for generating multiple-choice questions that target specific evaluation dimensions, integrating both automatic filtering and manual verification processes. Multiple-choice questions with groundtruth options derived from human annotation enables an objective and efficient assessment of model performance, eliminating the need for human or GPT intervention during evaluation. We further evaluate the performance of 18 models across all 12 dimensions, covering both the spatial and temporal understanding. By revealing the limitations of existing MLLMs through evaluation results, we aim for SEED-Bench to provide insights for motivating future research. We will launch and consistently maintain a leaderboard to provide a platform for the community to assess and investigate model capability.
    </details>
</div>
