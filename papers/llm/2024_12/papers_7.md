# llm - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19732v4">LLM as a Complementary Optimizer to Gradient Descent: A Case Study in Prompt Tuning</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Mastering a skill generally relies on both hands-on experience from doers and insightful, high-level guidance by mentors. Will this strategy also work well for solving complex non-convex optimization problems? Here, a common gradient-based optimizer acts like a disciplined doer, making locally optimal updates at each step. Large Language Models (LLMs) can also search for better solutions by inferring from natural language instructions, akin to a high-level mentor. In this paper, we show that these two participators are complementary to each other and can effectively collaborate as a combined optimization framework. The collaborative optimization is achieved by alternating between the gradient-based and LLM-based optimizers. We instruct LLMs to generate possibly improved solutions by taking parameter trajectories recorded during the previous stage of gradient-based optimization into account. Inferred results of LLMs are used as restarting points for the next stage of gradient optimization. We verify the effectiveness of this optimization framework on prompt tuning. By leveraging both the locally rigorous gradient-based optimizer and the high-level deductive LLM-based optimizer, the combined optimization method consistently yields improvements over competitive baselines on a variety of tasks. Our results demonstrate the synergistic effect of conventional gradient-based optimization and the inference ability of LLMs. The code is released at https://github.com/guozix/LLM-catalyst.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15903v2">LLM-Based Multi-Hop Question Answering with Knowledge Graph Integration in Evolving Environments</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      The important challenge of keeping knowledge in Large Language Models (LLMs) up-to-date has led to the development of various methods for incorporating new facts. However, existing methods for such knowledge editing still face difficulties with multi-hop questions that require accurate fact identification and sequential logical reasoning, particularly among numerous fact updates. To tackle these challenges, this paper introduces Graph Memory-based Editing for Large Language Models (GMeLLo), a straightforward and effective method that merges the explicit knowledge representation of Knowledge Graphs (KGs) with the linguistic flexibility of LLMs. Beyond merely leveraging LLMs for question answering, GMeLLo employs these models to convert free-form language into structured queries and fact triples, facilitating seamless interaction with KGs for rapid updates and precise multi-hop reasoning. Our results show that GMeLLo significantly surpasses current state-of-the-art (SOTA) knowledge editing methods in the multi-hop question answering benchmark, MQuAKE, especially in scenarios with extensive knowledge edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03359v1">WiS Platform: Enhancing Evaluation of LLM-Based Multi-Agent Systems Through Game-Based Analysis</a></div>
    <div class="paper-meta">
      📅 2024-12-04
    </div>
    <details class="paper-abstract">
      Recent advancements in autonomous multi-agent systems (MAS) based on large language models (LLMs) have enhanced the application scenarios and improved the capability of LLMs to handle complex tasks. Despite demonstrating effectiveness, existing studies still evidently struggle to evaluate, analysis, and reproducibility of LLM-based MAS. In this paper, to facilitate the research on LLM-based MAS, we introduce an open, scalable, and real-time updated platform for accessing and analyzing the LLM-based MAS based on the games Who is Spy?" (WiS). Our platform is featured with three main worths: (1) a unified model evaluate interface that supports models available on Hugging Face; (2) real-time updated leaderboard for model evaluation; (3) a comprehensive evaluation covering game-winning rates, attacking, defense strategies, and reasoning of LLMs. To rigorously test WiS, we conduct extensive experiments coverage of various open- and closed-source LLMs, we find that different agents exhibit distinct and intriguing behaviors in the game. The experimental results demonstrate the effectiveness and efficiency of our platform in evaluating LLM-based MAS. Our platform and its documentation are publicly available at \url{https://whoisspy.ai/}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03644v2">When LLMs Meet Cybersecurity: A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 We have updated the related papers up to Aug 31st, with 50+ new papers added
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) has opened new avenues across various fields, including cybersecurity, which faces an evolving threat landscape and demand for innovative technologies. Despite initial explorations into the application of LLMs in cybersecurity, there is a lack of a comprehensive overview of this research area. This paper addresses this gap by providing a systematic literature review, covering the analysis of over 300 works, encompassing 25 LLMs and more than 10 downstream scenarios. Our comprehensive overview addresses three key research questions: the construction of cybersecurity-oriented LLMs, the application of LLMs to various cybersecurity tasks, the challenges and further research in this area. This study aims to shed light on the extensive potential of LLMs in enhancing cybersecurity practices and serve as a valuable resource for applying LLMs in this field. We also maintain and regularly update a list of practical guides on LLMs for cybersecurity at https://github.com/tmylla/Awesome-LLM4Cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03612v1">Chatting with Logs: An exploratory study on Finetuning LLMs for LogQL</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 draft under submission at another venue
    </div>
    <details class="paper-abstract">
      Logging is a critical function in modern distributed applications, but the lack of standardization in log query languages and formats creates significant challenges. Developers currently must write ad hoc queries in platform-specific languages, requiring expertise in both the query language and application-specific log details -- an impractical expectation given the variety of platforms and volume of logs and applications. While generating these queries with large language models (LLMs) seems intuitive, we show that current LLMs struggle with log-specific query generation due to the lack of exposure to domain-specific knowledge. We propose a novel natural language (NL) interface to address these inconsistencies and aide log query generation, enabling developers to create queries in a target log query language by providing NL inputs. We further introduce ~\textbf{NL2QL}, a manually annotated, real-world dataset of natural language questions paired with corresponding LogQL queries spread across three log formats, to promote the training and evaluation of NL-to-loq query systems. Using NL2QL, we subsequently fine-tune and evaluate several state of the art LLMs, and demonstrate their improved capability to generate accurate LogQL queries. We perform further ablation studies to demonstrate the effect of additional training data, and the transferability across different log formats. In our experiments, we find up to 75\% improvement of finetuned models to generate LogQL queries compared to non finetuned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14845v2">AAVENUE: Detecting LLM Biases on NLU Tasks in AAVE via a Novel Benchmark</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Published at NLP4PI @ EMNLP 2024
    </div>
    <details class="paper-abstract">
      Detecting biases in natural language understanding (NLU) for African American Vernacular English (AAVE) is crucial to developing inclusive natural language processing (NLP) systems. To address dialect-induced performance discrepancies, we introduce AAVENUE ({AAVE} {N}atural Language {U}nderstanding {E}valuation), a benchmark for evaluating large language model (LLM) performance on NLU tasks in AAVE and Standard American English (SAE). AAVENUE builds upon and extends existing benchmarks like VALUE, replacing deterministic syntactic and morphological transformations with a more flexible methodology leveraging LLM-based translation with few-shot prompting, improving performance across our evaluation metrics when translating key tasks from the GLUE and SuperGLUE benchmarks. We compare AAVENUE and VALUE translations using five popular LLMs and a comprehensive set of metrics including fluency, BARTScore, quality, coherence, and understandability. Additionally, we recruit fluent AAVE speakers to validate our translations for authenticity. Our evaluations reveal that LLMs consistently perform better on SAE tasks than AAVE-translated versions, underscoring inherent biases and highlighting the need for more inclusive NLP models. We have open-sourced our source code on GitHub and created a website to showcase our work at https://aavenue.live.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03253v1">Alignment at Pre-training! Towards Native Alignment for Arabic LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Accepted to NeurIPS 2024 main conference. see https://github.com/FreedomIntelligence/AceGPT-v2
    </div>
    <details class="paper-abstract">
      The alignment of large language models (LLMs) is critical for developing effective and safe language models. Traditional approaches focus on aligning models during the instruction tuning or reinforcement learning stages, referred to in this paper as `post alignment'. We argue that alignment during the pre-training phase, which we term `native alignment', warrants investigation. Native alignment aims to prevent unaligned content from the beginning, rather than relying on post-hoc processing. This approach leverages extensively aligned pre-training data to enhance the effectiveness and usability of pre-trained models. Our study specifically explores the application of native alignment in the context of Arabic LLMs. We conduct comprehensive experiments and ablation studies to evaluate the impact of native alignment on model performance and alignment stability. Additionally, we release open-source Arabic LLMs that demonstrate state-of-the-art performance on various benchmarks, providing significant benefits to the Arabic LLM community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03248v1">AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enabled the creation of multi-modal LLMs that exhibit strong comprehension of visual data such as images and videos. However, these models usually rely on extensive visual tokens from visual encoders, leading to high computational demands, which limits their applicability in resource-constrained environments and for long-context tasks. In this work, we propose a training-free adaptive inference method for multi-modal LLMs that can accommodate a broad range of efficiency requirements with a minimum performance drop. Our method consists of a) iterative token merging based on embedding similarity before LLMs, and b) progressive token pruning within LLM layers based on multi-modal importance. With a minimalist design, our method can be applied to both video and image LLMs. Extensive experiments on diverse video and image benchmarks demonstrate that, our method substantially reduces computation load (e.g., a $\textbf{7-fold}$ reduction in FLOPs) while preserving the performance of video and image LLMs. Further, under a similar computational cost, our method outperforms the state-of-the-art methods in long video understanding (e.g., $\textbf{+4.6}$ on MLVU). Additionally, our in-depth analysis provides insights into token redundancy and LLM layer behaviors, offering guidance for future research in designing efficient multi-modal LLMs. Our code will be available at https://github.com/LaVi-Lab/AIM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02626v2">Time-Reversal Provides Unsupervised Feedback to LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Accepted as a spotlight in NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically trained to predict in the forward direction of time. However, recent works have shown that prompting these models to look back and critique their own generations can produce useful feedback. Motivated by this, we explore the question of whether LLMs can be empowered to think (predict and score) backwards to provide unsupervised feedback that complements forward LLMs. Towards this, we introduce Time Reversed Language Models (TRLMs), which can score and generate queries when conditioned on responses, effectively functioning in the reverse direction of time. Further, to effectively infer in the response to query direction, we pre-train and fine-tune a language model (TRLM-Ba) in the reverse token order from scratch. We show empirically (and theoretically in a stylized setting) that time-reversed models can indeed complement forward model predictions when used to score the query given response for re-ranking multiple forward generations. We obtain up to 5\% improvement on the widely used AlpacaEval Leaderboard over the competent baseline of best-of-N re-ranking using self log-perplexity scores. We further show that TRLM scoring outperforms conventional forward scoring of response given query, resulting in significant gains in applications such as citation generation and passage retrieval. We next leverage the generative ability of TRLM to augment or provide unsupervised feedback to input safety filters of LLMs, demonstrating a drastic reduction in false negative rate with negligible impact on false positive rates against several attacks published on the popular JailbreakBench leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03235v1">Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?</a></div>
    <div class="paper-meta">
      📅 2024-12-04
      | 💬 Accepted at the Safe Generative AI Workshop @ NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19318v2">Enhancing Trust in LLM-Generated Code Summaries with Calibrated Confidence Scores</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      A good summary can often be very useful during program comprehension. While a brief, fluent, and relevant summary can be helpful, it does require significant human effort to produce. Often, good summaries are unavailable in software projects, thus making maintenance more difficult. There has been a considerable body of research into automated AI-based methods, using Large Language models (LLMs), to generate summaries of code; there also has been quite a bit work on ways to measure the performance of such summarization methods, with special attention paid to how closely these AI-generated summaries resemble a summary a human might have produced. Measures such as BERTScore and BLEU have been suggested and evaluated with human-subject studies. However, LLM-produced summaries can be too long, irrelevant, etc: generally, too dissimilar to what a human might say. Given an LLM-produced code summary, how can we judge if a summary is good enough? Given some input source code, and an LLM-generated summary, existing approaches can help judge brevity, fluency and relevance; however, it's difficult to gauge whether an LLM-produced summary sufficiently resembles what a human might produce, without a "golden" human-produced summary to compare against. We study this resemblance question as a calibration problem: given just the summary from an LLM, can we compute a confidence measure, that provides a reliable indication of whether the summary sufficiently resembles what a human would have produced in this situation? We examine this question using several LLMs, for several languages, and in several different settings. Our investigation suggests approaches to provide reliable predictions of the likelihood that an LLM-generated summary would sufficiently resemble a summary a human might write for the same code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02906v1">Does Few-Shot Learning Help LLM Performance in Code Synthesis?</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant strides at code generation through improved model design, training, and chain-of-thought. However, prompt-level optimizations remain an important yet under-explored aspect of LLMs for coding. This work focuses on the few-shot examples present in most code generation prompts, offering a systematic study on whether few-shot examples improve LLM's coding capabilities, which few-shot examples have the largest impact, and how to select impactful examples. Our work offers 2 approaches for selecting few-shot examples, a model-free method, CODEEXEMPLAR-FREE, and a model-based method, CODEEXEMPLAR-BASED. The 2 methods offer a trade-off between improved performance and reliance on training data and interpretability. Both methods significantly improve CodeLlama's coding ability across the popular HumanEval+ coding benchmark. In summary, our work provides valuable insights into how to pick few-shot examples in code generation prompts to improve LLM code generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02883v1">TDD-Bench Verified: Can LLMs Generate Tests for Issues Before They Get Resolved?</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Test-driven development (TDD) is the practice of writing tests first and coding later, and the proponents of TDD expound its numerous benefits. For instance, given an issue on a source code repository, tests can clarify the desired behavior among stake-holders before anyone writes code for the agreed-upon fix. Although there has been a lot of work on automated test generation for the practice "write code first, test later", there has been little such automation for TDD. Ideally, tests for TDD should be fail-to-pass (i.e., fail before the issue is resolved and pass after) and have good adequacy with respect to covering the code changed during issue resolution. This paper introduces TDD-Bench Verified, a high-quality benchmark suite of 449 issues mined from real-world GitHub code repositories. The benchmark's evaluation harness runs only relevant tests in isolation for simple yet accurate coverage measurements, and the benchmark's dataset is filtered both by human judges and by execution in the harness. This paper also presents Auto-TDD, an LLM-based solution that takes as input an issue description and a codebase (prior to issue resolution) and returns as output a test that can be used to validate the changes made for resolving the issue. Our evaluation shows that Auto-TDD yields a better fail-to-pass rate than the strongest prior work while also yielding high coverage adequacy. Overall, we hope that this work helps make developers more productive at resolving issues while simultaneously leading to more robust fixes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14485v2">Mediating Modes of Thought: LLM's for design scripting</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 Published at ACADIA 2024
    </div>
    <details class="paper-abstract">
      Architects adopt visual scripting and parametric design tools to explore more expansive design spaces (Coates, 2010), refine their thinking about the geometric logic of their design (Woodbury, 2010), and overcome conventional software limitations (Burry, 2011). Despite two decades of effort to make design scripting more accessible, a disconnect between a designer's free ways of thinking and the rigidity of algorithms remains (Burry, 2011). Recent developments in Large Language Models (LLMs) suggest this might soon change, as LLMs encode a general understanding of human context and exhibit the capacity to produce geometric logic. This project speculates that if LLMs can effectively mediate between user intent and algorithms, they become a powerful tool to make scripting in design more widespread and fun. We explore if such systems can interpret natural language prompts to assemble geometric operations relevant to computational design scripting. In the system, multiple layers of LLM agents are configured with specific context to infer the user intent and construct a sequential logic. Given a user's high-level text prompt, a geometric description is created, distilled into a sequence of logic operations, and mapped to software-specific commands. The completed script is constructed in the user's visual programming interface. The system succeeds in generating complete visual scripts up to a certain complexity but fails beyond this complexity threshold. It shows how LLMs can make design scripting much more aligned with human creativity and thought. Future research should explore conversational interactions, expand to multimodal inputs and outputs, and assess the performance of these tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02868v1">A Novel Compact LLM Framework for Local, High-Privacy EHR Data Applications</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities in natural language processing, yet their use in sensitive domains like healthcare, particularly with Electronic Health Records (EHR), faces significant challenges due to privacy concerns and limited computational resources. This paper presents a compact LLM framework designed for local deployment in settings with strict privacy requirements and limited access to high-performance GPUs. We introduce a novel preprocessing technique that uses information extraction methods, e.g., regular expressions, to filter and emphasize critical information in clinical notes, enhancing the performance of smaller LLMs on EHR data. Our framework is evaluated using zero-shot and few-shot learning paradigms on both private and publicly available (MIMIC-IV) datasets, and we also compare its performance with fine-tuned LLMs on the MIMIC-IV dataset. The results demonstrate that our preprocessing approach significantly boosts the prediction accuracy of smaller LLMs, making them suitable for high-privacy, resource-constrained applications. This study offers valuable insights into optimizing LLM performance for sensitive, data-intensive tasks while addressing computational and privacy limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02816v1">Unleashing GHOST: An LLM-Powered Framework for Automated Hardware Trojan Design</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Traditionally, inserting realistic Hardware Trojans (HTs) into complex hardware systems has been a time-consuming and manual process, requiring comprehensive knowledge of the design and navigating intricate Hardware Description Language (HDL) codebases. Machine Learning (ML)-based approaches have attempted to automate this process but often face challenges such as the need for extensive training data, long learning times, and limited generalizability across diverse hardware design landscapes. This paper addresses these challenges by proposing GHOST (Generator for Hardware-Oriented Stealthy Trojans), an automated attack framework that leverages Large Language Models (LLMs) for rapid HT generation and insertion. Our study evaluates three state-of-the-art LLMs - GPT-4, Gemini-1.5-pro, and Llama-3-70B - across three hardware designs: SRAM, AES, and UART. According to our evaluations, GPT-4 demonstrates superior performance, with 88.88% of HT insertion attempts successfully generating functional and synthesizable HTs. This study also highlights the security risks posed by LLM-generated HTs, showing that 100% of GHOST-generated synthesizable HTs evaded detection by an ML-based HT detection tool. These results underscore the urgent need for advanced detection and prevention mechanisms in hardware security to address the emerging threat of LLM-generated HTs. The GHOST HT benchmarks are available at: https://github.com/HSTRG1/GHOSTbenchmarks.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15175v2">Can Open-source LLMs Enhance Data Synthesis for Toxic Detection?: An Experimental Study</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Effective toxic content detection relies heavily on high-quality and diverse data, which serves as the foundation for robust content moderation models. This study explores the potential of open-source LLMs for harmful data synthesis, utilizing prompt engineering and fine-tuning techniques to enhance data quality and diversity. In a two-stage evaluation, we first examine the capabilities of six open-source LLMs in generating harmful data across multiple datasets using prompt engineering. In the second stage, we fine-tune these models to improve data generation while addressing challenges such as hallucination, data duplication, and overfitting. Our findings reveal that Mistral excels in generating high-quality and diverse harmful data with minimal hallucination. Furthermore, fine-tuning enhances data quality, offering scalable and cost-effective solutions for augmenting datasets for specific toxic content detection tasks. These results emphasize the significance of data synthesis in building robust, standalone detection models and highlight the potential of open-source LLMs to advance smaller downstream content moderation systems. We implemented this approach in real-world industrial settings, demonstrating the feasibility and efficiency of fine-tuned open-source LLMs for harmful data synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14774v3">Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      We introduce Instruct-SkillMix, an automated approach for creating diverse, high quality SFT data. The Instruct-SkillMix pipeline involves two stages, each leveraging an existing powerful LLM: (1) Skill extraction: uses the LLM to extract core "skills" for instruction-following, either from existing datasets, or by directly prompting the model; (2) Data generation: uses the powerful LLM to generate (instruction, response) data that exhibit a randomly chosen pair of these skills. Here, the use of random skill combinations promotes diversity and difficulty. Vanilla SFT (i.e., no PPO, DPO, or RL methods) on data generated from Instruct-SkillMix leads to strong gains on instruction following benchmarks such as AlpacaEval 2.0, MT-Bench, and WildBench. With just $4$K examples, LLaMA-3-8B-Base achieves 42.76% length-controlled win rate on AlpacaEval 2.0. To our knowledge, this achieves state-of-the-art performance among all models that have only undergone SFT (no RL methods) and competes with proprietary models such as Claude 3 Opus and LLaMA-3.1-405B-Instruct. Ablation studies also suggest plausible reasons for why creating open instruction-tuning datasets via naive crowd-sourcing has proved difficult. Introducing low quality answers ("shirkers") in $20\%$ of Instruct-SkillMix examples causes performance to plummet, sometimes catastrophically. The Instruct-SkillMix pipeline is flexible and is adaptable to other settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01297v3">When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 TACL 2024
    </div>
    <details class="paper-abstract">
      Self-correction is an approach to improving responses from large language models (LLMs) by refining the responses using LLMs during inference. Prior work has proposed various self-correction frameworks using different sources of feedback, including self-evaluation and external feedback. However, there is still no consensus on the question of when LLMs can correct their own mistakes, as recent studies also report negative results. In this work, we critically survey broad papers and discuss the conditions required for successful self-correction. We first find that prior studies often do not define their research questions in detail and involve impractical frameworks or unfair evaluations that over-evaluate self-correction. To tackle these issues, we categorize research questions in self-correction research and provide a checklist for designing appropriate experiments. Our critical survey based on the newly categorized research questions shows that (1) no prior work demonstrates successful self-correction with feedback from prompted LLMs, except for studies in tasks that are exceptionally suited for self-correction, (2) self-correction works well in tasks that can use reliable external feedback, and (3) large-scale fine-tuning enables self-correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02764v1">Drawing Pandas: A Benchmark for LLMs in Generating Plotting Code</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      This paper introduces the human-curated PandasPlotBench dataset, designed to evaluate language models' effectiveness as assistants in visual data exploration. Our benchmark focuses on generating code for visualizing tabular data - such as a Pandas DataFrame - based on natural language instructions, complementing current evaluation tools and expanding their scope. The dataset includes 175 unique tasks. Our experiments assess several leading Large Language Models (LLMs) across three visualization libraries: Matplotlib, Seaborn, and Plotly. We show that the shortening of tasks has a minimal effect on plotting capabilities, allowing for the user interface that accommodates concise user input without sacrificing functionality or accuracy. Another of our findings reveals that while LLMs perform well with popular libraries like Matplotlib and Seaborn, challenges persist with Plotly, highlighting areas for improvement. We hope that the modular design of our benchmark will broaden the current studies on generating visualizations. Our benchmark is available online: https://huggingface.co/datasets/JetBrains-Research/plot_bench. The code for running the benchmark is also available: https://github.com/JetBrains-Research/PandasPlotBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02735v1">CPP-UT-Bench: Can LLMs Write Complex Unit Tests in C++?</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      We introduce CPP-UT-Bench, a benchmark dataset to measure C++ unit test generation capability of a large language model (LLM). CPP-UT-Bench aims to reflect a broad and diverse set of C++ codebases found in the real world. The dataset includes 2,653 {code, unit test} pairs drawn from 14 different opensource C++ codebases spanned across nine diverse domains including machine learning, software testing, parsing, standard input-output, data engineering, logging, complete expression evaluation, key value storage, and server protocols. We demonstrated the effectiveness of CPP-UT-Bench as a benchmark dataset through extensive experiments in in-context learning, parameter-efficient fine-tuning (PEFT), and full-parameter fine-tuning. We also discussed the challenges of the dataset compilation and insights we learned from in-context learning and fine-tuning experiments. Besides the CPP-UT-Bench dataset and data compilation code, we are also offering the fine-tuned model weights for further research. For nine out of ten experiments, our fine-tuned LLMs outperformed the corresponding base models by an average of more than 70%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02655v1">LLM-Enhanced Path Planning: Safe and Efficient Autonomous Navigation with Instructional Inputs</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Autonomous navigation guided by natural language instructions is essential for improving human-robot interaction and enabling complex operations in dynamic environments. While large language models (LLMs) are not inherently designed for planning, they can significantly enhance planning efficiency by providing guidance and informing constraints to ensure safety. This paper introduces a planning framework that integrates LLMs with 2D occupancy grid maps and natural language commands to improve spatial reasoning and task execution in resource-limited settings. By decomposing high-level commands and real-time environmental data, the system generates structured navigation plans for pick-and-place tasks, including obstacle avoidance, goal prioritization, and adaptive behaviors. The framework dynamically recalculates paths to address environmental changes and aligns with implicit social norms for seamless human-robot interaction. Our results demonstrates the potential of LLMs to design context-aware system to enhance navigation efficiency and safety in industrial and dynamic environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02611v1">AV-Odyssey Bench: Can Your Multimodal LLMs Really Understand Audio-Visual Information?</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 Project page: https://av-odyssey.github.io/
    </div>
    <details class="paper-abstract">
      Recently, multimodal large language models (MLLMs), such as GPT-4o, Gemini 1.5 Pro, and Reka Core, have expanded their capabilities to include vision and audio modalities. While these models demonstrate impressive performance across a wide range of audio-visual applications, our proposed DeafTest reveals that MLLMs often struggle with simple tasks humans find trivial: 1) determining which of two sounds is louder, and 2) determining which of two sounds has a higher pitch. Motivated by these observations, we introduce AV-Odyssey Bench, a comprehensive audio-visual benchmark designed to assess whether those MLLMs can truly understand the audio-visual information. This benchmark encompasses 4,555 carefully crafted problems, each incorporating text, visual, and audio components. To successfully infer answers, models must effectively leverage clues from both visual and audio inputs. To ensure precise and objective evaluation of MLLM responses, we have structured the questions as multiple-choice, eliminating the need for human evaluation or LLM-assisted assessment. We benchmark a series of closed-source and open-source models and summarize the observations. By revealing the limitations of current models, we aim to provide useful insight for future dataset collection and model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02594v1">PrefixLLM: LLM-aided Prefix Circuit Design</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      Prefix circuits are fundamental components in digital adders, widely used in digital systems due to their efficiency in calculating carry signals. Synthesizing prefix circuits with minimized area and delay is crucial for enhancing the performance of modern computing systems. Recently, large language models (LLMs) have demonstrated a surprising ability to perform text generation tasks. We propose PrefixLLM, that leverages LLMs for prefix circuit synthesis. PrefixLLM transforms the prefix circuit synthesis task into a structured text generation problem, termed the Structured Prefix Circuit Representation (SPCR), and introduces an iterative framework to automatically and accurately generate valid SPCRs. We further present a design space exploration (DSE) framework that uses LLMs to iteratively search for area and delay optimized prefix circuits. Compared to state-of-the-art, PrefixLLM can reduce the area by 3.70% under the same delay constraint. This work highlights the use of LLMs in the synthesis of arithmetic circuits, which can be transformed into the structured text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10583v2">Personalization of Code Readability Evaluation Based on LLM Using Collaborative Filtering</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 2 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      Code readability is an important indicator of software maintenance as it can significantly impact maintenance efforts. Recently, LLM (large language models) have been utilized for code readability evaluation. However, readability evaluation differs among developers, so personalization of the evaluation by LLM is needed. This study proposes a method which calibrates the evaluation, using collaborative filtering. Our preliminary analysis suggested that the method effectively enhances the accuracy of the readability evaluation using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02252v1">Compressing KV Cache for Long-Context LLM Inference with Inter-Layer Attention Similarity</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      The increasing context window size in Large Language Models (LLMs), such as the GPT and LLaMA series, has improved their ability to tackle complex, long-text tasks, but at the cost of inference efficiency, particularly regarding memory and computational complexity. Existing methods, including selective token retention and window-based attention, improve efficiency but risk discarding important tokens needed for future text generation. In this paper, we propose an approach that enhances LLM efficiency without token loss by reducing the memory and computational load of less important tokens, rather than discarding them.We address two challenges: 1) investigating the distribution of important tokens in the context, discovering recent tokens are more important than distant tokens in context, and 2) optimizing resources for distant tokens by sharing attention scores across layers. The experiments show that our method saves $35\%$ KV cache without compromising the performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02228v1">BANER: Boundary-Aware LLMs for Few-Shot Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 Appear on COLING 2025
    </div>
    <details class="paper-abstract">
      Despite the recent success of two-stage prototypical networks in few-shot named entity recognition (NER), challenges such as over/under-detected false spans in the span detection stage and unaligned entity prototypes in the type classification stage persist. Additionally, LLMs have not proven to be effective few-shot information extractors in general. In this paper, we propose an approach called Boundary-Aware LLMs for Few-Shot Named Entity Recognition to address these issues. We introduce a boundary-aware contrastive learning strategy to enhance the LLM's ability to perceive entity boundaries for generalized entity spans. Additionally, we utilize LoRAHub to align information from the target domain to the source domain, thereby enhancing adaptive cross-domain classification capabilities. Extensive experiments across various benchmarks demonstrate that our framework outperforms prior methods, validating its effectiveness. In particular, the proposed strategies demonstrate effectiveness across a range of LLM architectures. The code and data are released on https://github.com/UESTC-GQJ/BANER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12361v3">Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Agents powered by large language models have shown remarkable abilities in solving complex tasks. However, most agent systems remain reactive, limiting their effectiveness in scenarios requiring foresight and autonomous decision-making. In this paper, we tackle the challenge of developing proactive agents capable of anticipating and initiating tasks without explicit human instructions. We propose a novel data-driven approach for this problem. Firstly, we collect real-world human activities to generate proactive task predictions. These predictions are then labeled by human annotators as either accepted or rejected. The labeled data is used to train a reward model that simulates human judgment and serves as an automatic evaluator of the proactiveness of LLM agents. Building on this, we develop a comprehensive data generation pipeline to create a diverse dataset, ProactiveBench, containing 6,790 events. Finally, we demonstrate that fine-tuning models with the proposed ProactiveBench can significantly elicit the proactiveness of LLM agents. Experimental results show that our fine-tuned model achieves an F1-Score of 66.47% in proactively offering assistance, outperforming all open-source and close-source models. These results highlight the potential of our method in creating more proactive and effective agent systems, paving the way for future advancements in human-agent collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00648v2">DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 24 pages, 38 figures, source code \url{https://github.com/JingyangXiang/DFRot}
    </div>
    <details class="paper-abstract">
      Rotating the activation and weight matrices to reduce the influence of outliers in large language models (LLMs) has recently attracted significant attention, particularly in the context of model quantization. Prior studies have shown that in low-precision quantization scenarios, such as 4-bit weights and 4-bit activations (W4A4), randomized Hadamard transforms can achieve significantly higher accuracy than randomized orthogonal transforms. Notably, the reason behind this phenomena remains unknown. In this paper, we find that these transformations show substantial improvement in eliminating outliers for common tokens and achieve similar quantization error. The primary reason for the accuracy difference lies in the fact that randomized Hadamard transforms can slightly reduce the quantization error for tokens with massive activations while randomized orthogonal transforms increase the quantization error. Due to the extreme rarity of these tokens and their critical impact on model accuracy, we consider this a long-tail optimization problem, and therefore construct a simple yet effective method: a weighted loss function. Additionally, we propose an optimization strategy for the rotation matrix that involves alternating optimization of quantization parameters while employing orthogonal Procrustes transforms to refine the rotation matrix. This makes the distribution of the rotated activation values more conducive to quantization, especially for tokens with massive activations. Our method enhances the Rotated LLMs by achieving dual free, Outlier-Free and Massive Activation-Free, dubbed as DFRot. Extensive experiments demonstrate the effectiveness and efficiency of DFRot. By tuning the rotation matrix using just a single sample, DFRot achieves a perplexity improvement of 0.25 and 0.21 on W4A4KV4 and W4A4KV16, respectively, for LLaMA3-8B, a model known for its quantization challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04504v1">Multi-Bin Batching for Increasing LLM Inference Throughput</a></div>
    <div class="paper-meta">
      📅 2024-12-03
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in popularity for their diverse capabilities, improving the efficiency of their inference systems has become increasingly critical. Batching LLM requests is a critical step in scheduling the inference jobs on servers (e.g. GPUs), enabling the system to maximize throughput by allowing multiple requests to be processed in parallel. However, requests often have varying generation lengths, causing resource underutilization, as hardware must wait for the longest-running request in the batch to complete before moving to the next batch. We formalize this problem from a queueing-theoretic perspective, and aim to design a control policy which is throughput-optimal. We propose Multi-Bin Batching, a simple yet effective method that can provably improve LLM inference throughput by grouping requests with similar (predicted) execution times into predetermined bins. Through a combination of theoretical analysis and experiments, including real-world LLM inference scenarios, we demonstrate significant throughput gains compared to standard batching approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02113v1">Trust & Safety of LLMs and LLMs in Trust & Safety</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\ By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety. This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18908v2">DuetML: Human-LLM Collaborative Machine Learning Framework for Non-Expert Users</a></div>
    <div class="paper-meta">
      📅 2024-12-03
      | 💬 22 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Machine learning (ML) models have significantly impacted various domains in our everyday lives. While large language models (LLMs) offer intuitive interfaces and versatility, task-specific ML models remain valuable for their efficiency and focused performance in specialized tasks. However, developing these models requires technical expertise, making it particularly challenging for non-expert users to customize them for their unique needs. Although interactive machine learning (IML) aims to democratize ML development through user-friendly interfaces, users struggle to translate their requirements into appropriate ML tasks. We propose human-LLM collaborative ML as a new paradigm bridging human-driven IML and machine-driven LLM approaches. To realize this vision, we introduce DuetML, a framework that integrates multimodal LLMs (MLLMs) as interactive agents collaborating with users throughout the ML process. Our system carefully balances MLLM capabilities with user agency by implementing both reactive and proactive interactions between users and MLLM agents. Through a comparative user study, we demonstrate that DuetML enables non-expert users to define training data that better aligns with target tasks without increasing cognitive load, while offering opportunities for deeper engagement with ML task formulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10794v3">Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Accepted by EMNLP 2024 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are susceptible to a type of attack known as jailbreaking, which misleads LLMs to output harmful contents. Although there are diverse jailbreak attack strategies, there is no unified understanding on why some methods succeed and others fail. This paper explores the behavior of harmful and harmless prompts in the LLM's representation space to investigate the intrinsic properties of successful jailbreak attacks. We hypothesize that successful attacks share some similar properties: They are effective in moving the representation of the harmful prompt towards the direction to the harmless prompts. We leverage hidden representations into the objective of existing jailbreak attacks to move the attacks along the acceptance direction, and conduct experiments to validate the above hypothesis using the proposed objective. We hope this study provides new insights into understanding how LLMs understand harmfulness information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13060v2">AERO: Softmax-Only LLMs for Efficient Private Inference</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 40 pages, 21 figures, and 9 tables
    </div>
    <details class="paper-abstract">
      The pervasiveness of proprietary language models has raised privacy concerns for users' sensitive data, emphasizing the need for private inference (PI), where inference is performed directly on encrypted inputs. However, current PI methods face prohibitively higher communication and latency overheads, primarily due to nonlinear operations. In this paper, we present a comprehensive analysis to understand the role of nonlinearities in transformer-based decoder-only language models. We introduce AERO, a four-step architectural optimization framework that refines the existing LLM architecture for efficient PI by systematically removing nonlinearities such as LayerNorm and GELU and reducing FLOPs counts. For the first time, we propose a Softmax-only architecture with significantly fewer FLOPs tailored for efficient PI. Furthermore, we devise a novel entropy regularization technique to improve the performance of Softmax-only models. AERO achieves up to 4.23$\times$ communication and 1.94$\times$ latency reduction. We validate the effectiveness of AERO by benchmarking it against the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03597v1">The Vulnerability of Language Model Benchmarks: Do They Accurately Reflect True LLM Performance?</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      The pursuit of leaderboard rankings in Large Language Models (LLMs) has created a fundamental paradox: models excel at standardized tests while failing to demonstrate genuine language understanding and adaptability. Our systematic analysis of NLP evaluation frameworks reveals pervasive vulnerabilities across the evaluation spectrum, from basic metrics to complex benchmarks like GLUE and MMLU. These vulnerabilities manifest through benchmark exploitation, dataset contamination, and evaluation bias, creating a false perception of progress in language understanding capabilities. Through extensive review of contemporary evaluation approaches, we identify significant limitations in static benchmark designs, human evaluation protocols, and LLM-as-judge frameworks, all of which compromise the reliability of current performance assessments. As LLM capabilities evolve and existing benchmarks become redundant, we lay the groundwork for new evaluation methods that resist manipulation, minimize data contamination, and assess domain-specific tasks. This requires frameworks that are adapted dynamically, addressing current limitations and providing a more accurate reflection of LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01928v1">MALT: Improving Reasoning with Multi-Agent LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Preliminary work
    </div>
    <details class="paper-abstract">
      Enabling effective collaboration among LLMs is a crucial step toward developing autonomous systems capable of solving complex problems. While LLMs are typically used as single-model generators, where humans critique and refine their outputs, the potential for jointly-trained collaborative models remains largely unexplored. Despite promising results in multi-agent communication and debate settings, little progress has been made in training models to work together on tasks. In this paper, we present a first step toward "Multi-agent LLM training" (MALT) on reasoning problems. Our approach employs a sequential multi-agent setup with heterogeneous LLMs assigned specialized roles: a generator, verifier, and refinement model iteratively solving problems. We propose a trajectory-expansion-based synthetic data generation process and a credit assignment strategy driven by joint outcome based rewards. This enables our post-training setup to utilize both positive and negative trajectories to autonomously improve each model's specialized capabilities as part of a joint sequential system. We evaluate our approach across MATH, GSM8k, and CQA, where MALT on Llama 3.1 8B models achieves relative improvements of 14.14%, 7.12%, and 9.40% respectively over the same baseline model. This demonstrates an early advance in multi-agent cooperative capabilities for performance on mathematical and common sense reasoning questions. More generally, our work provides a concrete direction for research around multi-agent LLM training approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01778v1">HackSynth: LLM Agent and Evaluation Framework for Autonomous Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 16 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We introduce HackSynth, a novel Large Language Model (LLM)-based agent capable of autonomous penetration testing. HackSynth's dual-module architecture includes a Planner and a Summarizer, which enable it to generate commands and process feedback iteratively. To benchmark HackSynth, we propose two new Capture The Flag (CTF)-based benchmark sets utilizing the popular platforms PicoCTF and OverTheWire. These benchmarks include two hundred challenges across diverse domains and difficulties, providing a standardized framework for evaluating LLM-based penetration testing agents. Based on these benchmarks, extensive experiments are presented, analyzing the core parameters of HackSynth, including creativity (temperature and top-p) and token utilization. Multiple open source and proprietary LLMs were used to measure the agent's capabilities. The experiments show that the agent performed best with the GPT-4o model, better than what the GPT-4o's system card suggests. We also discuss the safety and predictability of HackSynth's actions. Our findings indicate the potential of LLM-based agents in advancing autonomous penetration testing and the importance of robust safeguards. HackSynth and the benchmarks are publicly available to foster research on autonomous cybersecurity solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01694v1">Unlocking Video-LLM via Agent-of-Thoughts Distillation</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      This paper tackles the problem of video question answering (VideoQA), a task that often requires multi-step reasoning and a profound understanding of spatial-temporal dynamics. While large video-language models perform well on benchmarks, they often lack explainability and spatial-temporal grounding. In this paper, we propose Agent-of-Thoughts Distillation (AoTD), a method that enhances models by incorporating automatically generated Chain-of-Thoughts (CoTs) into the instruction-tuning process. Specifically, we leverage an agent-based system to decompose complex questions into sub-tasks, and address them with specialized vision models, the intermediate results are then treated as reasoning chains. We also introduce a verification mechanism using a large language model (LLM) to ensure the reliability of generated CoTs. Extensive experiments demonstrate that AoTD improves the performance on multiple-choice and open-ended benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01661v1">R-Bot: An LLM-based Query Rewrite System</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Query rewrite is essential for optimizing SQL queries to improve their execution efficiency without changing their results. Traditionally, this task has been tackled through heuristic and learning-based methods, each with its limitations in terms of inferior quality and low robustness. Recent advancements in LLMs offer a new paradigm by leveraging their superior natural language and code comprehension abilities. Despite their potential, directly applying LLMs like GPT-4 has faced challenges due to problems such as hallucinations, where the model might generate inaccurate or irrelevant results. To address this, we propose R-Bot, an LLM-based query rewrite system with a systematic approach. We first design a multi-source rewrite evidence preparation pipeline to generate query rewrite evidences for guiding LLMs to avoid hallucinations. We then propose a hybrid structure-semantics retrieval method that combines structural and semantic analysis to retrieve the most relevant rewrite evidences for effectively answering an online query. We next propose a step-by-step LLM rewrite method that iteratively leverages the retrieved evidences to select and arrange rewrite rules with self-reflection. We conduct comprehensive experiments on widely used benchmarks, and demonstrate the superior performance of our system, R-Bot, surpassing state-of-the-art query rewrite methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01617v1">If Eleanor Rigby Had Met ChatGPT: A Study on Loneliness in a Post-LLM World</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Loneliness, or the lack of fulfilling relationships, significantly impacts a person's mental and physical well-being and is prevalent worldwide. Previous research suggests that large language models (LLMs) may help mitigate loneliness. However, we argue that the use of widespread LLMs like ChatGPT is more prevalent--and riskier, as they are not designed for this purpose. To explore this, we analysed user interactions with ChatGPT, particularly those outside of its marketed use as task-oriented assistant. In dialogues classified as lonely, users frequently (37%) sought advice or validation, and received good engagement. However, ChatGPT failed in sensitive scenarios, like responding appropriately to suicidal ideation or trauma. We also observed a 35% higher incidence of toxic content, with women being 22 times more likely to be targeted than men. Our findings underscore ethical and legal questions about this technology, and note risks like radicalisation or further isolation. We conclude with recommendations for research and industry to address loneliness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01605v1">Medchain: Bridging the Gap Between LLM Agents and Clinical Practice through Interactive Sequential Benchmarking</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Clinical decision making (CDM) is a complex, dynamic process crucial to healthcare delivery, yet it remains a significant challenge for artificial intelligence systems. While Large Language Model (LLM)-based agents have been tested on general medical knowledge using licensing exams and knowledge question-answering tasks, their performance in the CDM in real-world scenarios is limited due to the lack of comprehensive testing datasets that mirror actual medical practice. To address this gap, we present MedChain, a dataset of 12,163 clinical cases that covers five key stages of clinical workflow. MedChain distinguishes itself from existing benchmarks with three key features of real-world clinical practice: personalization, interactivity, and sequentiality. Further, to tackle real-world CDM challenges, we also propose MedChain-Agent, an AI system that integrates a feedback mechanism and a MCase-RAG module to learn from previous cases and adapt its responses. MedChain-Agent demonstrates remarkable adaptability in gathering information dynamically and handling sequential clinical tasks, significantly outperforming existing approaches. The relevant dataset and code will be released upon acceptance of this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19655v2">Truth or Mirage? Towards End-to-End Factuality Evaluation with LLM-Oasis</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 15 pages. To be submitted to CL journal
    </div>
    <details class="paper-abstract">
      After the introduction of Large Language Models (LLMs), there have been substantial improvements in the performance of Natural Language Generation (NLG) tasks, including Text Summarization and Machine Translation. However, LLMs still produce outputs containing hallucinations, that is, content not grounded in factual information. Therefore, developing methods to assess the factuality of LLMs has become urgent. Indeed, resources for factuality evaluation have recently emerged. Although challenging, these resources face one or more of the following limitations: (i) they are tailored to a specific task or domain; (ii) they are limited in size, thereby preventing the training of new factuality evaluators; (iii) they are designed for simpler verification tasks, such as claim verification. To address these issues, we introduce LLM-Oasis, to the best of our knowledge the largest resource for training end-to-end factuality evaluators. LLM-Oasis is constructed by extracting claims from Wikipedia, falsifying a subset of these claims, and generating pairs of factual and unfactual texts. We then rely on human annotators to both validate the quality of our dataset and to create a gold standard test set for benchmarking factuality evaluation systems. Our experiments demonstrate that LLM-Oasis presents a significant challenge for state-of-the-art LLMs, with GPT-4o achieving up to 60% accuracy in our proposed end-to-end factuality evaluation task, highlighting its potential to drive future research in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01523v1">Data-Centric and Heterogeneity-Adaptive Sequence Parallelism for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Extending the context length (i.e., the maximum supported sequence length) of LLMs is of paramount significance. To facilitate long context training of LLMs, sequence parallelism has emerged as an essential technique, which scatters each input sequence across multiple devices and necessitates communication to process the sequence. In essence, existing sequence parallelism methods assume homogeneous sequence lengths (i.e., all input sequences are equal in length) and therefore leverages a single, static scattering strategy for all input sequences. However, in reality, the sequence lengths in LLM training corpora exhibit substantial variability, often following a long-tail distribution, which leads to workload heterogeneity. In this paper, we show that employing a single, static strategy results in inefficiency and resource under-utilization, highlighting the need for adaptive approaches to handle the heterogeneous workloads across sequences. To address this, we propose a heterogeneity-adaptive sequence parallelism method. For each training step, our approach captures the variability in sequence lengths and assigns the optimal combination of scattering strategies based on workload characteristics. We model this problem as a linear programming optimization and design an efficient and effective solver to find the optimal solution. Furthermore, we implement our method in a high-performance system that supports adaptive parallelization in distributed LLM training. Experimental results demonstrate that our system outperforms state-of-the-art training frameworks by up to 1.98x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01447v1">PLD+: Accelerating LLM inference by leveraging Language Model Artifacts</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      To reduce the latency associated with autoretrogressive LLM inference, speculative decoding has emerged as a novel decoding paradigm, where future tokens are drafted and verified in parallel. However, the practical deployment of speculative decoding is hindered by its requirements for additional computational resources and fine-tuning, which limits its out-of-the-box usability. To address these challenges, we present PLD+, a suite of novel algorithms developed to accelerate the inference process of LLMs, particularly for input-guided tasks. These tasks, which include code editing, text editing, summarization, etc., often feature outputs with substantial overlap with their inputs-an attribute PLD+ is designed to exploit. PLD+ also leverages the artifacts (attention and hidden states) generated during inference to accelerate inference speed. We test our approach on five input-guided tasks and through extensive experiments we find that PLD+ outperforms all tuning-free approaches. In the greedy setting, it even outperforms the state-of-the-art tuning-dependent approach EAGLE on four of the tasks. (by a margin of upto 2.31 in terms of avg. speedup). Our approach is tuning free, does not require any additional compute and can easily be used for accelerating inference of any LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01380v1">Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Main Text: 10 pages, 11 figures. Appendix: 3 pages, 3 figures
    </div>
    <details class="paper-abstract">
      While mobile devices provide ever more compute power, improvements in DRAM bandwidth are much slower. This is unfortunate for large language model (LLM) token generation, which is heavily memory-bound. Previous work has proposed to leverage natural dynamic activation sparsity in ReLU-activated LLMs to reduce effective DRAM bandwidth per token. However, more recent LLMs use SwiGLU instead of ReLU, which result in little inherent sparsity. While SwiGLU activations can be pruned based on magnitude, the resulting sparsity patterns are difficult to predict, rendering previous approaches ineffective. To circumvent this issue, our work introduces Dynamic Input Pruning (DIP): a predictor-free dynamic sparsification approach, which preserves accuracy with minimal fine-tuning. DIP can further use lightweight LoRA adapters to regain some performance lost during sparsification. Lastly, we describe a novel cache-aware masking strategy, which considers the cache state and activation magnitude to further increase cache hit rate, improving LLM token rate on mobile devices. DIP outperforms other methods in terms of accuracy, memory and throughput trade-offs across simulated hardware settings. On Phi-3-Medium, DIP achieves a 46% reduction in memory and 40% increase in throughput with $<$ 0.1 loss in perplexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14708v2">Understanding LLM Embeddings for Regression</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 16 pages, 13 figures
    </div>
    <details class="paper-abstract">
      With the rise of large language models (LLMs) for flexibly processing information as strings, a natural application is regression, specifically by preprocessing string representations into LLM embeddings as downstream features for metric prediction. In this paper, we provide one of the first comprehensive investigations into embedding-based regression and demonstrate that LLM embeddings as features can be better for high-dimensional regression tasks than using traditional feature engineering. This regression performance can be explained in part due to LLM embeddings over numeric data inherently preserving Lipschitz continuity over the feature space. Furthermore, we quantify the contribution of different model effects, most notably model size and language understanding, which we find surprisingly do not always improve regression performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01330v1">The "LLM World of Words" English free association norms generated by large language models</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 16 pages, 11 figures, associated Github page with dataset available at: https://github.com/LLMWorldOfWords/LWOW
    </div>
    <details class="paper-abstract">
      Free associations have been extensively used in cognitive psychology and linguistics for studying how conceptual knowledge is organized. Recently, the potential of applying a similar approach for investigating the knowledge encoded in LLMs has emerged, specifically as a method for investigating LLM biases. However, the absence of large-scale LLM-generated free association norms that are comparable with human-generated norms is an obstacle to this new research direction. To address this limitation, we create a new dataset of LLM-generated free association norms modeled after the "Small World of Words" (SWOW) human-generated norms consisting of approximately 12,000 cue words. We prompt three LLMs, namely Mistral, Llama3, and Haiku, with the same cues as those in the SWOW norms to generate three novel comparable datasets, the "LLM World of Words" (LWOW). Using both SWOW and LWOW norms, we construct cognitive network models of semantic memory that represent the conceptual knowledge possessed by humans and LLMs. We demonstrate how these datasets can be used for investigating implicit biases in humans and LLMs, such as the harmful gender stereotypes that are prevalent both in society and LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01230v1">GraphOTTER: Evolving LLM-based Graph Reasoning for Complex Table Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 COLING 2025, code is available at https://github.com/JDing0521/GraphOTTER
    </div>
    <details class="paper-abstract">
      Complex Table Question Answering involves providing accurate answers to specific questions based on intricate tables that exhibit complex layouts and flexible header locations. Despite considerable progress having been made in the LLM era, the reasoning processes of existing methods are often implicit, feeding the entire table into prompts, making it difficult to effectively filter out irrelevant information in the table. To this end, we propose GraphOTTER that explicitly establishes the reasoning process to pinpoint the correct answers. In particular, GraphOTTER leverages a graph-based representation, transforming the complex table into an undirected graph. It then conducts step-by-step reasoning on the graph, with each step guided by a set of pre-defined intermediate reasoning actions. As such, it constructs a clear reasoning path and effectively identifies the answer to a given question. Comprehensive experiments on two benchmark datasets and two LLM backbones demonstrate the effectiveness of GraphOTTER. Further analysis indicates that its success may be attributed to the ability to efficiently filter out irrelevant information, thereby focusing the reasoning process on the most pertinent data. Our code and experimental datasets are available at \url{https://github.com/JDing0521/GraphOTTER}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18363v2">ChatRex: Taming Multimodal LLM for Joint Perception and Understanding</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 35 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Perception and understanding are two pillars of computer vision. While multimodal large language models (MLLM) have demonstrated remarkable visual understanding capabilities, they arguably lack accurate perception abilities, e.g. the stage-of-the-art model Qwen2-VL only achieves a 43.9 recall rate on the COCO dataset, limiting many tasks requiring the combination of perception and understanding. In this work, we aim to bridge this perception gap from both model designing and data development perspectives. We first introduce ChatRex, an MLLM with a decoupled perception design. Instead of having the LLM directly predict box coordinates, we feed the output boxes from a universal proposal network into the LLM, allowing it to output the corresponding box indices to represent its detection results, turning the regression task into a retrieval-based task that LLM handles more proficiently. From the data perspective, we build a fully automated data engine and construct the Rexverse-2M dataset which possesses multiple granularities to support the joint training of perception and understanding. After standard two-stage training, ChatRex demonstrates strong perception capabilities while preserving multimodal understanding performance. The combination of these two capabilities simultaneously unlocks many attractive applications, demonstrating the complementary roles of both perception and understanding in MLLM. Code is available at \url{https://github.com/IDEA-Research/ChatRex}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19951v2">T2Vid: Translating Long Text into Multi-Image is the Catalyst for Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project page: https://github.com/xjtupanda/T2Vid
    </div>
    <details class="paper-abstract">
      The success of Multimodal Large Language Models (MLLMs) in the image domain has garnered wide attention from the research community. Drawing on previous successful experiences, researchers have recently explored extending the success to the video understanding realms. Apart from training from scratch, an efficient way is to utilize the pre-trained image-LLMs, leading to two mainstream approaches, i.e. zero-shot inference and further fine-tuning with video data. In this work, our study of these approaches harvests an effective data augmentation method. We first make a deeper inspection of the zero-shot inference way and identify two limitations, i.e. limited generalization and lack of temporal understanding capabilities. Thus, we further investigate the fine-tuning approach and find a low learning efficiency when simply using all the video data samples, which can be attributed to a lack of instruction diversity. Aiming at this issue, we develop a method called T2Vid to synthesize video-like samples to enrich the instruction diversity in the training corpus. Integrating these data enables a simple and efficient training scheme, which achieves performance comparable to or even superior to using full video datasets by training with just 15% the sample size. Meanwhile, we find that the proposed scheme can boost the performance of long video understanding without training with long video samples. We hope our study will spark more thinking about using MLLMs for video understanding and curation of high-quality data. The code is released at https://github.com/xjtupanda/T2Vid.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11285v2">Self and Cross-Model Distillation for LLMs: Effective Methods for Refusal Pattern Alignment</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 The method used in the paper has obvious problems and ambiguities. The security enhancement method we used cannot be considered distillation, but it is described as distillation in the paper, and the experiment lacks comparison and baseline, which has been criticized by many peers. In order to avoid further dissemination, we have decided to withdraw the paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) like OpenAI's GPT series, Anthropic's Claude, and Meta's LLaMa have shown remarkable capabilities in text generation. However, their susceptibility to toxic prompts presents significant security challenges. This paper investigates alignment techniques, including Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), to mitigate these risks. We conduct an empirical study on refusal patterns across nine LLMs, revealing that models with uniform refusal patterns, such as Claude3, exhibit higher security. Based on these findings, we propose self-distilling and cross-model distilling methods to enhance LLM security. Our results show that these methods significantly improve refusal rates and reduce unsafe content, with cross-model distilling achieving refusal rates close to Claude3's 94.51%. These findings underscore the potential of distillation-based alignment in securing LLMs against toxic prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01072v1">When Fine-Tuning LLMs Meets Data Privacy: An Empirical Study of Federated Learning in LLM-Based Program Repair</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Software systems have been evolving rapidly and inevitably introducing bugs at an increasing rate, leading to significant losses in resources consumed by software maintenance. Recently, large language models (LLMs) have demonstrated remarkable potential in enhancing software development and maintenance practices, particularly in automated program repair (APR) with improved accuracy and efficiency of bug fixing. However, LLM-based APR heavily relies on high-quality code repositories. A larger portion of existing code repositories are for private use and proprietary assets from various industries, reflecting more diversity and nuances in the data since real-world industries often have more extensive software development practices, which cannot be covered by merely public datasets. Therefore, utilizing private datasets shows significant potential in enhancing software development and maintenance. However, obtaining such data from various industries is hindered by data privacy concerns, as companies are reluctant to share their codebases. To address the gap, we investigate the use of federated learning as a privacy-preserving approach that enables private entities to fine-tune LLMs on proprietary and decentralized data, facilitating the collaboration between clients to fully utilize their data to help enhance software development and maintenance. Our evaluation reveals that federated fine-tuning can effectively enhance program repair capabilities. Notably, the impact of heterogeneous code on LLM fine-tuning is negligible, indicating that real-world industries can benefit from collaborative development regardless of diverse data distributions. Furthermore, each type of federated algorithm exhibits unique strengths across different LLMs, suggesting that fine-tuning for program repair can be enhanced by tailoring the optimization process to specific characteristics of different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17912v2">Can LLMs plan paths in the real world?</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly integrate into vehicle navigation systems, understanding their path-planning capability is crucial. We tested three LLMs through six real-world path-planning scenarios in various settings and with various difficulties. Our experiments showed that all LLMs made numerous errors in all scenarios, revealing that they are unreliable path planners. We suggest that future work focus on implementing mechanisms for reality checks, enhancing model transparency, and developing smaller models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02326v2">Evaluating LLMs for Hardware Design and Test</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated capabilities for producing code in Hardware Description Languages (HDLs). However, most of the focus remains on their abilities to write functional code, not test code. The hardware design process consists of both design and test, and so eschewing validation and verification leaves considerable potential benefit unexplored, given that a design and test framework may allow for progress towards full automation of the digital design pipeline. In this work, we perform one of the first studies exploring how a LLM can both design and test hardware modules from provided specifications. Using a suite of 8 representative benchmarks, we examined the capabilities and limitations of the state-of-the-art conversational LLMs when producing Verilog for functional and verification purposes. We taped out the benchmarks on a Skywater 130nm shuttle and received the functional chip.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01042v1">TruncFormer: Private LLM Inference Using Only Truncations</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Private inference (PI) serves an important role in guaranteeing the privacy of user data when interfacing with proprietary machine learning models such as LLMs. However, PI remains practically intractable due to the massive latency costs associated with nonlinear functions present in LLMs. Existing works have focused on improving latency of specific LLM nonlinearities (such as the Softmax, or the GeLU) via approximations. However, new types of nonlinearities are regularly introduced with new LLM architectures, and this has led to a constant game of catch-up where PI researchers attempt to optimize the newest nonlinear function. We introduce TruncFormer, a framework for taking any LLM and transforming it into a plaintext emulation of PI. Our framework leverages the fact that nonlinearities in LLMs are differentiable and can be accurately approximated with a sequence of additions, multiplications, and truncations. Further, we decouple the add/multiply and truncation operations, and statically determine where truncations should be inserted based on a given field size and input representation size. This leads to latency improvements over existing cryptographic protocols that enforce truncation after every multiplication operation. We open source our code for community use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01033v1">SAUP: Situation Awareness Uncertainty Propagation on LLM Agent</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) integrated into multistep agent systems enable complex decision-making processes across various applications. However, their outputs often lack reliability, making uncertainty estimation crucial. Existing uncertainty estimation methods primarily focus on final-step outputs, which fail to account for cumulative uncertainty over the multistep decision-making process and the dynamic interactions between agents and their environments. To address these limitations, we propose SAUP (Situation Awareness Uncertainty Propagation), a novel framework that propagates uncertainty through each step of an LLM-based agent's reasoning process. SAUP incorporates situational awareness by assigning situational weights to each step's uncertainty during the propagation. Our method, compatible with various one-step uncertainty estimation techniques, provides a comprehensive and accurate uncertainty measure. Extensive experiments on benchmark datasets demonstrate that SAUP significantly outperforms existing state-of-the-art methods, achieving up to 20% improvement in AUROC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01020v1">AI Benchmarks and Datasets for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 November 2024 v1.0
    </div>
    <details class="paper-abstract">
      LLMs demand significant computational resources for both pre-training and fine-tuning, requiring distributed computing capabilities due to their large model sizes \cite{sastry2024computing}. Their complex architecture poses challenges throughout the entire AI lifecycle, from data collection to deployment and monitoring \cite{OECD_AIlifecycle}. Addressing critical AI system challenges, such as explainability, corrigibility, interpretability, and hallucination, necessitates a systematic methodology and rigorous benchmarking \cite{guldimann2024complai}. To effectively improve AI systems, we must precisely identify systemic vulnerabilities through quantitative evaluation, bolstering system trustworthiness. The enactment of the EU AI Act \cite{EUAIAct} by the European Parliament on March 13, 2024, establishing the first comprehensive EU-wide requirements for the development, deployment, and use of AI systems, further underscores the importance of tools and methodologies such as Z-Inspection. It highlights the need to enrich this methodology with practical benchmarks to effectively address the technical challenges posed by AI systems. To this end, we have launched a project that is part of the AI Safety Bulgaria initiatives \cite{AI_Safety_Bulgaria}, aimed at collecting and categorizing AI benchmarks. This will enable practitioners to identify and utilize these benchmarks throughout the AI system lifecycle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00970v1">Generating AI Literacy MCQs: A Multi-Agent LLM Approach</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI) is transforming society, making it crucial to prepare the next generation through AI literacy in K-12 education. However, scalable and reliable AI literacy materials and assessment resources are lacking. To address this gap, our study presents a novel approach to generating multiple-choice questions (MCQs) for AI literacy assessments. Our method utilizes large language models (LLMs) to automatically generate scalable, high-quality assessment questions. These questions align with user-provided learning objectives, grade levels, and Bloom's Taxonomy levels. We introduce an iterative workflow incorporating LLM-powered critique agents to ensure the generated questions meet pedagogical standards. In the preliminary evaluation, experts expressed strong interest in using the LLM-generated MCQs, indicating that this system could enrich existing AI literacy materials and provide a valuable addition to the toolkit of K-12 educators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00967v1">Linear Probe Penalties Reduce LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2024-12-01
      | 💬 20 pages, 15 figures, NeurIPS 2024 Workshop Socially Responsible Language Modelling Research (SoLaR)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often sycophantic, prioritizing agreement with their users over accurate or objective statements. This problematic behavior becomes more pronounced during reinforcement learning from human feedback (RLHF), an LLM fine-tuning stage intended to align model outputs with human values. Instead of increasing accuracy and reliability, the reward model learned from RLHF often rewards sycophancy. We develop a linear probing method to identify and penalize markers of sycophancy within the reward model, producing rewards that discourage sycophantic behavior. Our experiments show that constructing and optimizing against this surrogate reward function reduces sycophantic behavior in multiple open-source LLMs. Our results suggest a generalizable methodology for reducing unwanted LLM behaviors that are not sufficiently disincentivized by RLHF fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00962v1">LLMs as mirrors of societal moral standards: reflection of cultural divergence and agreement across ethical topics</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly pivotal in various domains due the recent advancements in their performance capabilities. However, concerns persist regarding biases in LLMs, including gender, racial, and cultural biases derived from their training data. These biases raise critical questions about the ethical deployment and societal impact of LLMs. Acknowledging these concerns, this study investigates whether LLMs accurately reflect cross-cultural variations and similarities in moral perspectives. In assessing whether the chosen LLMs capture patterns of divergence and agreement on moral topics across cultures, three main methods are employed: (1) comparison of model-generated and survey-based moral score variances, (2) cluster alignment analysis to evaluate the correspondence between country clusters derived from model-generated moral scores and those derived from survey data, and (3) probing LLMs with direct comparative prompts. All three methods involve the use of systematic prompts and token pairs designed to assess how well LLMs understand and reflect cultural variations in moral attitudes. The findings of this study indicate overall variable and low performance in reflecting cross-cultural differences and similarities in moral values across the models tested, highlighting the necessity for improving models' accuracy in capturing these nuances effectively. The insights gained from this study aim to inform discussions on the ethical development and deployment of LLMs in global contexts, emphasizing the importance of mitigating biases and promoting fair representation across diverse cultural perspectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01444v2">No Size Fits All: The Perils and Pitfalls of Leveraging LLMs Vary with Company Size</a></div>
    <div class="paper-meta">
      📅 2024-12-01
      | 💬 COLING2025 Industry track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are playing a pivotal role in deploying strategic use cases across a range of organizations, from large pan-continental companies to emerging startups. The issues and challenges involved in the successful utilization of LLMs can vary significantly depending on the size of the organization. It is important to study and discuss these pertinent issues of LLM adaptation with a focus on the scale of the industrial concerns and brainstorm possible solutions and prospective directions. Such a study has not been prominently featured in the current research literature. In this study, we adopt a threefold strategy: first, we conduct a case study with industry practitioners to formulate the key research questions; second, we examine existing industrial publications to address these questions; and finally, we provide a practical guide for industries to utilize LLMs more efficiently. We release the GitHub\footnote{\url{https://github.com/vinayakcse/IndustrialLLMsPapers}} repository with the most recent papers in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00846v1">Improving Multimodal LLMs Ability In Geometry Problem Solving, Reasoning, And Multistep Scoring</a></div>
    <div class="paper-meta">
      📅 2024-12-01
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      This paper presents GPSM4K, a comprehensive geometry multimodal dataset tailored to augment the problem-solving capabilities of Large Vision Language Models (LVLMs). GPSM4K encompasses 2157 multimodal question-answer pairs manually extracted from mathematics textbooks spanning grades 7-12 and is further augmented to 5340 problems, consisting of both numerical and theorem-proving questions. In contrast to PGPS9k, Geometry3K, and Geo170K which feature only objective-type questions, GPSM4K offers detailed step-by-step solutions in a consistent format, facilitating a comprehensive evaluation of problem-solving approaches. This dataset serves as an excellent benchmark for assessing the geometric reasoning capabilities of LVLMs. Evaluation of our test set shows that there is scope for improvement needed in open-source language models in geometry problem-solving. Finetuning on our training set increases the geometry problem-solving capabilities of models. Further, We also evaluate the effectiveness of techniques such as image captioning and Retrieval Augmentation generation (RAG) on model performance. We leveraged LLM to automate the task of final answer evaluation by providing ground truth and predicted solutions. This research will help to assess and improve the geometric reasoning capabilities of LVLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00804v1">Does chat change LLM's mind? Impact of Conversation on Psychological States of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-01
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The recent growth of large language models (LLMs) has enabled more authentic, human-centered interactions through multi-agent systems. However, investigation into how conversations affect the psychological states of LLMs is limited, despite the impact of these states on the usability of LLM-based systems. In this study, we explored whether psychological states change during multi-agent interactions, focusing on the effects of conversation depth, topic, and speaker. We experimentally investigated the behavior of 10 LLMs in open-domain conversations. We employed 14 questionnaires and a topic-analysis method to examine the behavior of LLMs across four aspects: personality, interpersonal relationships, motivation, and emotion. The results revealed distinct psychological trends influenced by conversation depth and topic, with significant variations observed between different LLM families and parameter sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00765v1">SelfPrompt: Autonomously Evaluating LLM Robustness via Domain-Constrained Knowledge Guidelines and Refined Adversarial Prompts</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      Traditional methods for evaluating the robustness of large language models (LLMs) often rely on standardized benchmarks, which can escalate costs and limit evaluations across varied domains. This paper introduces a novel framework designed to autonomously evaluate the robustness of LLMs by incorporating refined adversarial prompts and domain-constrained knowledge guidelines in the form of knowledge graphs. Our method systematically generates descriptive sentences from domain-constrained knowledge graph triplets to formulate adversarial prompts, enhancing the relevance and challenge of the evaluation. These prompts, generated by the LLM itself and tailored to evaluate its own robustness, undergo a rigorous filtering and refinement process, ensuring that only those with high textual fluency and semantic fidelity are used. This self-evaluation mechanism allows the LLM to evaluate its robustness without the need for external benchmarks. We assess the effectiveness of our framework through extensive testing on both proprietary models like ChatGPT and open-source models such as Llama-3.1, Phi-3, and Mistral. Results confirm that our approach not only reduces dependency on conventional data but also provides a targeted and efficient means of evaluating LLM robustness in constrained domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00726v1">Free and Customizable Code Documentation with LLMs: A Fine-Tuning Approach</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      Automated documentation of programming source code is a challenging task with significant practical and scientific implications for the developer community. We present a large language model (LLM)-based application that developers can use as a support tool to generate basic documentation for any publicly available repository. Over the last decade, several papers have been written on generating documentation for source code using neural network architectures. With the recent advancements in LLM technology, some open-source applications have been developed to address this problem. However, these applications typically rely on the OpenAI APIs, which incur substantial financial costs, particularly for large repositories. Moreover, none of these open-source applications offer a fine-tuned model or features to enable users to fine-tune. Additionally, finding suitable data for fine-tuning is often challenging. Our application addresses these issues which is available at https://pypi.org/project/readme-ready/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14491v3">A Survey on Human-Centric LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) and their capacity to simulate human cognition and behavior has given rise to LLM-based frameworks and tools that are evaluated and applied based on their ability to perform tasks traditionally performed by humans, namely those involving cognition, decision-making, and social interaction. This survey provides a comprehensive examination of such human-centric LLM capabilities, focusing on their performance in both individual tasks (where an LLM acts as a stand-in for a single human) and collective tasks (where multiple LLMs coordinate to mimic group dynamics). We first evaluate LLM competencies across key areas including reasoning, perception, and social cognition, comparing their abilities to human-like skills. Then, we explore real-world applications of LLMs in human-centric domains such as behavioral science, political science, and sociology, assessing their effectiveness in replicating human behaviors and interactions. Finally, we identify challenges and future research directions, such as improving LLM adaptability, emotional intelligence, and cultural sensitivity, while addressing inherent biases and enhancing frameworks for human-AI collaboration. This survey aims to provide a foundational understanding of LLMs from a human-centric perspective, offering insights into their current capabilities and potential for future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05315v1">Text Is Not All You Need: Multimodal Prompting Helps LLMs Understand Humor</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive natural language understanding capabilities across various text-based tasks, understanding humor has remained a persistent challenge. Humor is frequently multimodal, relying on phonetic ambiguity, rhythm and timing to convey meaning. In this study, we explore a simple multimodal prompting approach to humor understanding and explanation. We present an LLM with both the text and the spoken form of a joke, generated using an off-the-shelf text-to-speech (TTS) system. Using multimodal cues improves the explanations of humor compared to textual prompts across all tested datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12279v3">HouseLLM: LLM-Assisted Two-Phase Text-to-Floorplan Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      This paper proposes a two-phase text-to-floorplan generation method, which guides a Large Language Model (LLM) to generate an initial layout (Layout-LLM) and refines them into the final floorplans through conditional diffusion model. We incorporate a Chain-of-Thought approach to prompt the LLM based on user text specifications, enabling a more user-friendly and intuitive house layout design. This method allows users to describe their needs in natural language, enhancing accessibility and providing clearer geometric constraints. The final floorplans generated by Layout-LLM through conditional diffusion refinement are more accurate and better meet user requirements. Experimental results demonstrate that our approach achieves state-of-the-art performance across all metrics, validating its effectiveness in practical home design applications. We plan to release our code for public use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00631v1">ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      Instruction tuning has underscored the significant potential of large language models (LLMs) in producing more human-controllable and effective outputs in various domains. In this work, we focus on the data selection problem for task-specific instruction tuning of LLMs. Prevailing methods primarily rely on the crafted similarity metrics to select training data that aligns with the test data distribution. The goal is to minimize instruction tuning loss on the test data, ultimately improving performance on the target task. However, it has been widely observed that instruction tuning loss (i.e., cross-entropy loss for next token prediction) in LLMs often fails to exhibit a monotonic relationship with actual task performance. This misalignment undermines the effectiveness of current data selection methods for task-specific instruction tuning. To address this issue, we introduce ROSE, a novel Reward-Oriented inStruction data sElection method which leverages pairwise preference loss as a reward signal to optimize data selection for task-specific instruction tuning. Specifically, ROSE adapts an influence formulation to approximate the influence of training data points relative to a few-shot preference validation set to select the most task-related training data points. Experimental results show that by selecting just 5% of the training data using ROSE, our approach can achieve competitive results compared to fine-tuning with the full training dataset, and it surpasses other state-of-the-art data selection methods for task-specific instruction tuning. Our qualitative analysis further confirms the robust generalizability of our method across multiple benchmark datasets and diverse model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00621v1">Exposing LLM Vulnerabilities: Adversarial Scam Detection and Performance</a></div>
    <div class="paper-meta">
      📅 2024-12-01
      | 💬 4 pages, 2024 IEEE International Conference on Big Data workshop BigEACPS 2024
    </div>
    <details class="paper-abstract">
      Can we trust Large Language Models (LLMs) to accurately predict scam? This paper investigates the vulnerabilities of LLMs when facing adversarial scam messages for the task of scam detection. We addressed this issue by creating a comprehensive dataset with fine-grained labels of scam messages, including both original and adversarial scam messages. The dataset extended traditional binary classes for the scam detection task into more nuanced scam types. Our analysis showed how adversarial examples took advantage of vulnerabilities of a LLM, leading to high misclassification rate. We evaluated the performance of LLMs on these adversarial scam messages and proposed strategies to improve their robustness.
    </details>
</div>
