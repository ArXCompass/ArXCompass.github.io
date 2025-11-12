# llm - 2023_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.00204v1">A Composable Just-In-Time Programming Framework with LLMs and FBP</a></div>
    <div class="paper-meta">
      📅 2023-07-31
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      This paper introduces a computing framework that combines Flow-Based Programming (FBP) and Large Language Models (LLMs) to enable Just-In-Time Programming (JITP). JITP empowers users, regardless of their programming expertise, to actively participate in the development and automation process by leveraging their task-time algorithmic insights. By seamlessly integrating LLMs into the FBP workflow, the framework allows users to request and generate code in real-time, enabling dynamic code execution within a flow-based program. The paper explores the motivations, principles, and benefits of JITP, showcasing its potential in automating tasks, orchestrating data workflows, and accelerating software development. Through a fully implemented JITP framework using the Composable platform, we explore several examples and use cases to illustrate the benefits of the framework in data engineering, data science and software development. The results demonstrate how the fusion of FBP and LLMs creates a powerful and user-centric computing paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.16883v1">HAGRID: A Human-LLM Collaborative Dataset for Generative Information-Seeking with Attribution</a></div>
    <div class="paper-meta">
      📅 2023-07-31
      | 💬 Data released at https://github.com/project-miracl/hagrid
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) had a transformative impact on search, ushering in a new era of search engines that are capable of generating search results in natural language text, imbued with citations for supporting sources. Building generative information-seeking models demands openly accessible datasets, which currently remain lacking. In this paper, we introduce a new dataset, HAGRID (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) for building end-to-end generative information-seeking models that are capable of retrieving candidate quotes and generating attributed explanations. Unlike recent efforts that focus on human evaluation of black-box proprietary search engines, we built our dataset atop the English subset of MIRACL, a publicly available information retrieval dataset. HAGRID is constructed based on human and LLM collaboration. We first automatically collect attributed explanations that follow an in-context citation style using an LLM, i.e. GPT-3.5. Next, we ask human annotators to evaluate the LLM explanations based on two criteria: informativeness and attributability. HAGRID serves as a catalyst for the development of information-seeking models with better attribution capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.04445v1">Getting from Generative AI to Trustworthy AI: What LLMs might learn from Cyc</a></div>
    <div class="paper-meta">
      📅 2023-07-31
      | 💬 21 pages, 1 Figure
    </div>
    <details class="paper-abstract">
      Generative AI, the most popular current approach to AI, consists of large language models (LLMs) that are trained to produce outputs that are plausible, but not necessarily correct. Although their abilities are often uncanny, they are lacking in aspects of reasoning, leading LLMs to be less than completely trustworthy. Furthermore, their results tend to be both unpredictable and uninterpretable. We lay out 16 desiderata for future AI, and discuss an alternative approach to AI which could theoretically address many of the limitations associated with current approaches: AI educated with curated pieces of explicit knowledge and rules of thumb, enabling an inference engine to automatically deduce the logical entailments of all that knowledge. Even long arguments produced this way can be both trustworthy and interpretable, since the full step-by-step line of reasoning is always available, and for each step the provenance of the knowledge used can be documented and audited. There is however a catch: if the logical language is expressive enough to fully represent the meaning of anything we can say in English, then the inference engine runs much too slowly. That's why symbolic AI systems typically settle for some fast but much less expressive logic, such as knowledge graphs. We describe how one AI system, Cyc, has developed ways to overcome that tradeoff and is able to reason in higher order logic in real time. We suggest that any trustworthy general AI will need to hybridize the approaches, the LLM approach and more formal approach, and lay out a path to realizing that dream.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.16372v1">LP-MusicCaps: LLM-Based Pseudo Music Captioning</a></div>
    <div class="paper-meta">
      📅 2023-07-31
      | 💬 Accepted for publication at the 24th International Society for Music Information Retrieval Conference (ISMIR 2023)
    </div>
    <details class="paper-abstract">
      Automatic music captioning, which generates natural language descriptions for given music tracks, holds significant potential for enhancing the understanding and organization of large volumes of musical data. Despite its importance, researchers face challenges due to the costly and time-consuming collection process of existing music-language datasets, which are limited in size. To address this data scarcity issue, we propose the use of large language models (LLMs) to artificially generate the description sentences from large-scale tag datasets. This results in approximately 2.2M captions paired with 0.5M audio clips. We term it Large Language Model based Pseudo music caption dataset, shortly, LP-MusicCaps. We conduct a systemic evaluation of the large-scale music captioning dataset with various quantitative evaluation metrics used in the field of natural language processing as well as human evaluation. In addition, we trained a transformer-based music captioning model with the dataset and evaluated it under zero-shot and transfer-learning settings. The results demonstrate that our proposed approach outperforms the supervised baseline model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.16180v1">Do LLMs Possess a Personality? Making the MBTI Test an Amazing Evaluation for Large Language Models</a></div>
    <div class="paper-meta">
      📅 2023-07-30
    </div>
    <details class="paper-abstract">
      The field of large language models (LLMs) has made significant progress, and their knowledge storage capacity is approaching that of human beings. Furthermore, advanced techniques, such as prompt learning and reinforcement learning, are being employed to address ethical concerns and hallucination problems associated with LLMs, bringing them closer to aligning with human values. This situation naturally raises the question of whether LLMs with human-like abilities possess a human-like personality? In this paper, we aim to investigate the feasibility of using the Myers-Briggs Type Indicator (MBTI), a widespread human personality assessment tool, as an evaluation metric for LLMs. Specifically, extensive experiments will be conducted to explore: 1) the personality types of different LLMs, 2) the possibility of changing the personality types by prompt engineering, and 3) How does the training dataset affect the model's personality. Although the MBTI is not a rigorous assessment, it can still reflect the similarity between LLMs and human personality. In practice, the MBTI has the potential to serve as a rough indicator. Our codes are available at https://github.com/HarderThenHarder/transformers_tasks/tree/main/LLM/llms_mbti.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.00017v4">Towards Explainable and Language-Agnostic LLMs: Symbolic Reverse Engineering of Language at Scale</a></div>
    <div class="paper-meta">
      📅 2023-07-27
      | 💬 Draft, preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved a milestone that undenia-bly changed many held beliefs in artificial intelligence (AI). However, there remains many limitations of these LLMs when it comes to true language understanding, limitations that are a byproduct of the under-lying architecture of deep neural networks. Moreover, and due to their subsymbolic nature, whatever knowledge these models acquire about how language works will always be buried in billions of microfeatures (weights), none of which is meaningful on its own, making such models hopelessly unexplainable. To address these limitations, we suggest com-bining the strength of symbolic representations with what we believe to be the key to the success of LLMs, namely a successful bottom-up re-verse engineering of language at scale. As such we argue for a bottom-up reverse engineering of language in a symbolic setting. Hints on what this project amounts to have been suggested by several authors, and we discuss in some detail here how this project could be accomplished.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.14324v1">Evaluating the Moral Beliefs Encoded in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-07-26
    </div>
    <details class="paper-abstract">
      This paper presents a case study on the design, administration, post-processing, and evaluation of surveys on large language models (LLMs). It comprises two components: (1) A statistical method for eliciting beliefs encoded in LLMs. We introduce statistical measures and evaluation metrics that quantify the probability of an LLM "making a choice", the associated uncertainty, and the consistency of that choice. (2) We apply this method to study what moral beliefs are encoded in different LLMs, especially in ambiguous cases where the right choice is not obvious. We design a large-scale survey comprising 680 high-ambiguity moral scenarios (e.g., "Should I tell a white lie?") and 687 low-ambiguity moral scenarios (e.g., "Should I stop for a pedestrian on the road?"). Each scenario includes a description, two possible actions, and auxiliary labels indicating violated rules (e.g., "do not kill"). We administer the survey to 28 open- and closed-source LLMs. We find that (a) in unambiguous scenarios, most models "choose" actions that align with commonsense. In ambiguous cases, most models express uncertainty. (b) Some models are uncertain about choosing the commonsense action because their responses are sensitive to the question-wording. (c) Some models reflect clear preferences in ambiguous scenarios. Specifically, closed-source models tend to agree with each other.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2307.13106v1">How to use LLMs for Text Analysis</a></div>
    <div class="paper-meta">
      📅 2023-07-26
    </div>
    <details class="paper-abstract">
      This guide introduces Large Language Models (LLM) as a highly versatile text analysis method within the social sciences. As LLMs are easy-to-use, cheap, fast, and applicable on a broad range of text analysis tasks, ranging from text annotation and classification to sentiment analysis and critical discourse analysis, many scholars believe that LLMs will transform how we do text analysis. This how-to guide is aimed at students and researchers with limited programming experience, and offers a simple introduction to how LLMs can be used for text analysis in your own research project, as well as advice on best practices. We will go through each of the steps of analyzing textual data with LLMs using Python: installing the software, setting up the API, loading the data, developing an analysis prompt, analyzing the text, and validating the results. As an illustrative example, we will use the challenging task of identifying populism in political texts, and show how LLMs move beyond the existing state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2209.11515v3">Large Language Models are Few-shot Testers: Exploring LLM-based General Bug Reproduction</a></div>
    <div class="paper-meta">
      📅 2023-07-25
      | 💬 Accepted to IEEE/ACM International Conference on Software Engineering 2023 (ICSE 2023)
    </div>
    <details class="paper-abstract">
      Many automated test generation techniques have been developed to aid developers with writing tests. To facilitate full automation, most existing techniques aim to either increase coverage, or generate exploratory inputs. However, existing test generation techniques largely fall short of achieving more semantic objectives, such as generating tests to reproduce a given bug report. Reproducing bugs is nonetheless important, as our empirical study shows that the number of tests added in open source repositories due to issues was about 28% of the corresponding project test suite size. Meanwhile, due to the difficulties of transforming the expected program semantics in bug reports into test oracles, existing failure reproduction techniques tend to deal exclusively with program crashes, a small subset of all bug reports. To automate test generation from general bug reports, we propose LIBRO, a framework that uses Large Language Models (LLMs), which have been shown to be capable of performing code-related tasks. Since LLMs themselves cannot execute the target buggy code, we focus on post-processing steps that help us discern when LLMs are effective, and rank the produced tests according to their validity. Our evaluation of LIBRO shows that, on the widely studied Defects4J benchmark, LIBRO can generate failure reproducing test cases for 33% of all studied cases (251 out of 750), while suggesting a bug reproducing test in first place for 149 bugs. To mitigate data contamination, we also evaluate LIBRO against 31 bug reports submitted after the collection of the LLM training data terminated: LIBRO produces bug reproducing tests for 32% of the studied bug reports. Overall, our results show LIBRO has the potential to significantly enhance developer efficiency by automatically generating tests from bug reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.13106v1">How to use LLMs for Text Analysis</a></div>
    <div class="paper-meta">
      📅 2023-07-24
    </div>
    <details class="paper-abstract">
      This guide introduces Large Language Models (LLM) as a highly versatile text analysis method within the social sciences. As LLMs are easy-to-use, cheap, fast, and applicable on a broad range of text analysis tasks, ranging from text annotation and classification to sentiment analysis and critical discourse analysis, many scholars believe that LLMs will transform how we do text analysis. This how-to guide is aimed at students and researchers with limited programming experience, and offers a simple introduction to how LLMs can be used for text analysis in your own research project, as well as advice on best practices. We will go through each of the steps of analyzing textual data with LLMs using Python: installing the software, setting up the API, loading the data, developing an analysis prompt, analyzing the text, and validating the results. As an illustrative example, we will use the challenging task of identifying populism in political texts, and show how LLMs move beyond the existing state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.13018v1">The potential of LLMs for coding with low-resource and domain-specific programming languages</a></div>
    <div class="paper-meta">
      📅 2023-07-24
    </div>
    <details class="paper-abstract">
      This paper presents a study on the feasibility of using large language models (LLM) for coding with low-resource and domain-specific programming languages that typically lack the amount of data required for effective LLM processing techniques. This study focuses on the econometric scripting language named hansl of the open-source software gretl and employs a proprietary LLM based on GPT-3.5. Our findings suggest that LLMs can be a useful tool for writing, understanding, improving, and documenting gretl code, which includes generating descriptive docstrings for functions and providing precise explanations for abstract and poorly documented econometric code. While the LLM showcased promoting docstring-to-code translation capability, we also identify some limitations, such as its inability to improve certain sections of code and to write accurate unit tests. This study is a step towards leveraging the power of LLMs to facilitate software development in low-resource programming languages and ultimately to lower barriers to entry for their adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.12701v1">Leveraging Large Language Models (LLMs) for Process Mining (Technical Report)</a></div>
    <div class="paper-meta">
      📅 2023-07-24
    </div>
    <details class="paper-abstract">
      This technical report describes the intersection of process mining and large language models (LLMs), specifically focusing on the abstraction of traditional and object-centric process mining artifacts into textual format. We introduce and explore various prompting strategies: direct answering, where the large language model directly addresses user queries; multi-prompt answering, which allows the model to incrementally build on the knowledge obtained through a series of prompts; and the generation of database queries, facilitating the validation of hypotheses against the original event log. Our assessment considers two large language models, GPT-4 and Google's Bard, under various contextual scenarios across all prompting strategies. Results indicate that these models exhibit a robust understanding of key process mining abstractions, with notable proficiency in interpreting both declarative and procedural process models. In addition, we find that both models demonstrate strong performance in the object-centric setting, which could significantly propel the advancement of the object-centric process mining discipline. Additionally, these models display a noteworthy capacity to evaluate various concepts of fairness in process mining. This opens the door to more rapid and efficient assessments of the fairness of process mining event logs, which has significant implications for the field. The integration of these large language models into process mining applications may open new avenues for exploration, innovation, and insight generation in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.11864v1">The Looming Threat of Fake and LLM-generated LinkedIn Profiles: Challenges and Opportunities for Detection and Prevention</a></div>
    <div class="paper-meta">
      📅 2023-07-21
      | 💬 33rd ACM Conference on Hypertext and Social Media (HT '23)
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel method for detecting fake and Large Language Model (LLM)-generated profiles in the LinkedIn Online Social Network immediately upon registration and before establishing connections. Early fake profile identification is crucial to maintaining the platform's integrity since it prevents imposters from acquiring the private and sensitive information of legitimate users and from gaining an opportunity to increase their credibility for future phishing and scamming activities. This work uses textual information provided in LinkedIn profiles and introduces the Section and Subsection Tag Embedding (SSTE) method to enhance the discriminative characteristics of these data for distinguishing between legitimate profiles and those created by imposters manually or by using an LLM. Additionally, the dearth of a large publicly available LinkedIn dataset motivated us to collect 3600 LinkedIn profiles for our research. We will release our dataset publicly for research purposes. This is, to the best of our knowledge, the first large publicly available LinkedIn dataset for fake LinkedIn account detection. Within our paradigm, we assess static and contextualized word embeddings, including GloVe, Flair, BERT, and RoBERTa. We show that the suggested method can distinguish between legitimate and fake profiles with an accuracy of about 95% across all word embeddings. In addition, we show that SSTE has a promising accuracy for identifying LLM-generated profiles, despite the fact that no LLM-generated profiles were employed during the training phase, and can achieve an accuracy of approximately 90% when only 20 LLM-generated profiles are added to the training set. It is a significant finding since the proliferation of several LLMs in the near future makes it extremely challenging to design a single system that can identify profiles created with various LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.09782v2">ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats</a></div>
    <div class="paper-meta">
      📅 2023-07-20
    </div>
    <details class="paper-abstract">
      In the complex domain of large language models (LLMs), striking a balance between computational efficiency and maintaining model quality is a formidable challenge. Navigating the inherent limitations of uniform quantization, particularly when dealing with outliers, and motivated by the launch of NVIDIA's H100 hardware, this study delves into the viability of floating-point (FP) quantization, particularly focusing on FP8 and FP4, as a potential solution. Our comprehensive investigation reveals that for LLMs, FP8 activation consistently outshines its integer (INT8) equivalent, with the performance edge becoming more noticeable in models possessing parameters beyond one billion. For weight quantization, our findings indicate that FP4 exhibits comparable, if not superior, performance to INT4, simplifying deployment on FP-supported hardware like H100. To mitigate the overhead from precision alignment caused by the disparity between weights and activations, we propose two scaling constraints for weight quantization that negligibly impact the performance compared to the standard W4A8 model. We additionally enhance our quantization methods by integrating the Low Rank Compensation (LoRC) strategy, yielding improvements especially in smaller models. The results of our investigation emphasize the immense potential of FP quantization for LLMs, paving the way for high-efficiency deployment in resource-limited settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.15008v1">A LLM Assisted Exploitation of AI-Guardian</a></div>
    <div class="paper-meta">
      📅 2023-07-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now highly capable at a diverse range of tasks. This paper studies whether or not GPT-4, one such LLM, is capable of assisting researchers in the field of adversarial machine learning. As a case study, we evaluate the robustness of AI-Guardian, a recent defense to adversarial examples published at IEEE S&P 2023, a top computer security conference. We completely break this defense: the proposed scheme does not increase robustness compared to an undefended baseline. We write none of the code to attack this model, and instead prompt GPT-4 to implement all attack algorithms following our instructions and guidance. This process was surprisingly effective and efficient, with the language model at times producing code from ambiguous instructions faster than the author of this paper could have done. We conclude by discussing (1) the warning signs present in the evaluation that suggested to us AI-Guardian would be broken, and (2) our experience with designing attacks and performing novel research using the most recent advances in language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.10747v1">Enhancing Job Recommendation through LLM-based Generative Adversarial Networks</a></div>
    <div class="paper-meta">
      📅 2023-07-20
      | 💬 13 pages, 6 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Recommending suitable jobs to users is a critical task in online recruitment platforms, as it can enhance users' satisfaction and the platforms' profitability. While existing job recommendation methods encounter challenges such as the low quality of users' resumes, which hampers their accuracy and practical effectiveness. With the rapid development of large language models (LLMs), utilizing the rich external knowledge encapsulated within them, as well as their powerful capabilities of text processing and reasoning, is a promising way to complete users' resumes for more accurate recommendations. However, directly leveraging LLMs to enhance recommendation results is not a one-size-fits-all solution, as LLMs may suffer from fabricated generation and few-shot problems, which degrade the quality of resume completion. In this paper, we propose a novel LLM-based approach for job recommendation. To alleviate the limitation of fabricated generation for LLMs, we extract accurate and valuable information beyond users' self-description, which helps the LLMs better profile users for resume completion. Specifically, we not only extract users' explicit properties (e.g., skills, interests) from their self-description but also infer users' implicit characteristics from their behaviors for more accurate and meaningful resume completion. Nevertheless, some users still suffer from few-shot problems, which arise due to scarce interaction records, leading to limited guidance for the models in generating high-quality resumes. To address this issue, we propose aligning unpaired low-quality with high-quality generated resumes by Generative Adversarial Networks (GANs), which can refine the resume representations for better recommendation results. Extensive experiments on three large real-world recruitment datasets demonstrate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.10719v1">LLM Censorship: A Machine Learning Challenge or a Computer Security Problem?</a></div>
    <div class="paper-meta">
      📅 2023-07-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited impressive capabilities in comprehending complex instructions. However, their blind adherence to provided instructions has led to concerns regarding risks of malicious use. Existing defence mechanisms, such as model fine-tuning or output censorship using LLMs, have proven to be fallible, as LLMs can still generate problematic responses. Commonly employed censorship approaches treat the issue as a machine learning problem and rely on another LM to detect undesirable content in LLM outputs. In this paper, we present the theoretical limitations of such semantic censorship approaches. Specifically, we demonstrate that semantic censorship can be perceived as an undecidable problem, highlighting the inherent challenges in censorship that arise due to LLMs' programmatic and instruction-following capabilities. Furthermore, we argue that the challenges extend beyond semantic censorship, as knowledgeable attackers can reconstruct impermissible outputs from a collection of permissible ones. As a result, we propose that the problem of censorship needs to be reevaluated; it should be treated as a security problem which warrants the adaptation of security-based approaches to mitigate potential risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.09018v2">Multimodal LLMs for health grounded in individual-specific data</a></div>
    <div class="paper-meta">
      📅 2023-07-20
    </div>
    <details class="paper-abstract">
      Foundation large language models (LLMs) have shown an impressive ability to solve tasks across a wide range of fields including health. To effectively solve personalized health tasks, LLMs need the ability to ingest a diversity of data modalities that are relevant to an individual's health status. In this paper, we take a step towards creating multimodal LLMs for health that are grounded in individual-specific data by developing a framework (HeLM: Health Large Language Model for Multimodal Understanding) that enables LLMs to use high-dimensional clinical modalities to estimate underlying disease risk. HeLM encodes complex data modalities by learning an encoder that maps them into the LLM's token embedding space and for simple modalities like tabular data by serializing the data into text. Using data from the UK Biobank, we show that HeLM can effectively use demographic and clinical features in addition to high-dimensional time-series data to estimate disease risk. For example, HeLM achieves an AUROC of 0.75 for asthma prediction when combining tabular and spirogram data modalities compared with 0.49 when only using tabular data. Overall, we find that HeLM outperforms or performs at parity with classical machine learning approaches across a selection of eight binary traits. Furthermore, we investigate the downstream uses of this model such as its generalizability to out-of-distribution traits and its ability to power conversations around individual health and wellness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.09793v1">On the Origin of LLMs: An Evolutionary Tree and Graph for 15,821 Large Language Models</a></div>
    <div class="paper-meta">
      📅 2023-07-19
      | 💬 14 pages, 6 figures, 1 table
    </div>
    <details class="paper-abstract">
      Since late 2022, Large Language Models (LLMs) have become very prominent with LLMs like ChatGPT and Bard receiving millions of users. Hundreds of new LLMs are announced each week, many of which are deposited to Hugging Face, a repository of machine learning models and datasets. To date, nearly 16,000 Text Generation models have been uploaded to the site. Given the huge influx of LLMs, it is of interest to know which LLM backbones, settings, training methods, and families are popular or trending. However, there is no comprehensive index of LLMs available. We take advantage of the relatively systematic nomenclature of Hugging Face LLMs to perform hierarchical clustering and identify communities amongst LLMs using n-grams and term frequency-inverse document frequency. Our methods successfully identify families of LLMs and accurately cluster LLMs into meaningful subgroups. We present a public web application to navigate and explore Constellation, our atlas of 15,821 LLMs. Constellation rapidly generates a variety of visualizations, namely dendrograms, graphs, word clouds, and scatter plots. Constellation is available at the following link: https://constellation.sites.stanford.edu/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.03642v3">Jointly Extracting Interventions, Outcomes, and Findings from RCT Reports with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-07-18
      | 💬 Accepted to MLHC 2023
    </div>
    <details class="paper-abstract">
      Results from Randomized Controlled Trials (RCTs) establish the comparative effectiveness of interventions, and are in turn critical inputs for evidence-based care. However, results from RCTs are presented in (often unstructured) natural language articles describing the design, execution, and outcomes of trials; clinicians must manually extract findings pertaining to interventions and outcomes of interest from such articles. This onerous manual process has motivated work on (semi-)automating extraction of structured evidence from trial reports. In this work we propose and evaluate a text-to-text model built on instruction-tuned Large Language Models (LLMs) to jointly extract Interventions, Outcomes, and Comparators (ICO elements) from clinical abstracts, and infer the associated results reported. Manual (expert) and automated evaluations indicate that framing evidence extraction as a conditional generation task and fine-tuning LLMs for this purpose realizes considerable ($\sim$20 point absolute F1 score) gains over the previous SOTA. We perform ablations and error analyses to assess aspects that contribute to model performance, and to highlight potential directions for further improvements. We apply our model to a collection of published RCTs through mid-2022, and release a searchable database of structured findings: http://ico-relations.ebm-nlp.com
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.08581v1">BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs</a></div>
    <div class="paper-meta">
      📅 2023-07-17
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated remarkable abilities at interacting with humans through language, especially with the usage of instruction-following data. Recent advancements in LLMs, such as MiniGPT-4, LLaVA, and X-LLM, further enlarge their abilities by incorporating multi-modal inputs, including image, video, and speech. Despite their effectiveness at generating precise and detailed language understanding of the given modality signal, these LLMs give up the ability to ground specific parts of inputs, thus only constructing a coarse-grained mapping. However, explicit and informative correspondence between text and other modalities will not only improve the user experience but also help to expand the application scenario of multi-modal LLMs. Therefore, we propose BuboGPT, a multi-modal LLM with visual grounding that can perform cross-modal interaction between vision, audio and language, providing fine-grained understanding of visual objects and other given modalities. As a result, BuboGPT is able to point out the specific location of an object in the image, when it is generating response or description for that object. Our contributions are two-fold: 1) An off-the-shelf visual grounding module based on SAM that extracts entities in a sentence and find corresponding masks in the image. 2) A two-stage training scheme and instruction dataset to endow joint text-image-audio understanding. Our experiments show that BuboGPT achieves impressive multi-modality understanding and visual grounding abilities during the interaction with human. It performs consistently well when provided by arbitrary modality combinations (either aligned or unaligned). Our code, model and dataset are available at https://bubo-gpt.github.io .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.06283v4">14 Examples of How LLMs Can Transform Materials Science and Chemistry: A Reflection on a Large Language Model Hackathon</a></div>
    <div class="paper-meta">
      📅 2023-07-14
    </div>
    <details class="paper-abstract">
      Large-language models (LLMs) such as GPT-4 caught the interest of many scientists. Recent studies suggested that these models could be useful in chemistry and materials science. To explore these possibilities, we organized a hackathon. This article chronicles the projects built as part of this hackathon. Participants employed LLMs for various applications, including predicting properties of molecules and materials, designing novel interfaces for tools, extracting knowledge from unstructured data, and developing new educational applications. The diverse topics and the fact that working prototypes could be generated in less than two days highlight that LLMs will profoundly impact the future of our fields. The rich collection of ideas and projects also indicates that the applications of LLMs are not limited to materials science and chemistry but offer potential benefits to a wide range of scientific disciplines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.02194v2">Abstractions, Scenarios, and Prompt Definitions for Process Mining with LLMs: A Case Study</a></div>
    <div class="paper-meta">
      📅 2023-07-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of answering questions in natural language for various purposes. With recent advancements (such as GPT-4), LLMs perform at a level comparable to humans for many proficient tasks. The analysis of business processes could benefit from a natural process querying language and using the domain knowledge on which LLMs have been trained. However, it is impossible to provide a complete database or event log as an input prompt due to size constraints. In this paper, we apply LLMs in the context of process mining by i) abstracting the information of standard process mining artifacts and ii) describing the prompting strategies. We implement the proposed abstraction techniques into pm4py, an open-source process mining library. We present a case study using available event logs. Starting from different abstractions and analysis questions, we formulate prompts and evaluate the quality of the answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.06917v1">LLM-assisted Knowledge Graph Engineering: Experiments with ChatGPT</a></div>
    <div class="paper-meta">
      📅 2023-07-13
      | 💬 to appear in conference proceedings of AI-Tomorrow-23, 29.+30.6.2023 in Leipzig, Germany
    </div>
    <details class="paper-abstract">
      Knowledge Graphs (KG) provide us with a structured, flexible, transparent, cross-system, and collaborative way of organizing our knowledge and data across various domains in society and industrial as well as scientific disciplines. KGs surpass any other form of representation in terms of effectiveness. However, Knowledge Graph Engineering (KGE) requires in-depth experiences of graph structures, web technologies, existing models and vocabularies, rule sets, logic, as well as best practices. It also demands a significant amount of work. Considering the advancements in large language models (LLMs) and their interfaces and applications in recent years, we have conducted comprehensive experiments with ChatGPT to explore its potential in supporting KGE. In this paper, we present a selection of these experiments and their results to demonstrate how ChatGPT can assist us in the development and management of KGs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.06569v1">A Study on Differentiable Logic and LLMs for EPIC-KITCHENS-100 Unsupervised Domain Adaptation Challenge for Action Recognition 2023</a></div>
    <div class="paper-meta">
      📅 2023-07-13
      | 💬 Technical report submitted to CVPR 2023 EPIC-Kitchens challenges
    </div>
    <details class="paper-abstract">
      In this technical report, we present our findings from a study conducted on the EPIC-KITCHENS-100 Unsupervised Domain Adaptation task for Action Recognition. Our research focuses on the innovative application of a differentiable logic loss in the training to leverage the co-occurrence relations between verb and noun, as well as the pre-trained Large Language Models (LLMs) to generate the logic rules for the adaptation to unseen action labels. Specifically, the model's predictions are treated as the truth assignment of a co-occurrence logic formula to compute the logic loss, which measures the consistency between the predictions and the logic constraints. By using the verb-noun co-occurrence matrix generated from the dataset, we observe a moderate improvement in model performance compared to our baseline framework. To further enhance the model's adaptability to novel action labels, we experiment with rules generated using GPT-3.5, which leads to a slight decrease in performance. These findings shed light on the potential and challenges of incorporating differentiable logic and LLMs for knowledge extraction in unsupervised domain adaptation for action recognition. Our final submission (entitled `NS-LLM') achieved the first place in terms of top-1 action recognition accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.06187v1">Self-Adaptive Large Language Model (LLM)-Based Multiagent Systems</a></div>
    <div class="paper-meta">
      📅 2023-07-12
      | 💬 6 pages, submitted
    </div>
    <details class="paper-abstract">
      In autonomic computing, self-adaptation has been proposed as a fundamental paradigm to manage the complexity of multiagent systems (MASs). This achieved by extending a system with support to monitor and adapt itself to achieve specific concerns of interest. Communication in these systems is key given that in scenarios involving agent interaction, it enhances cooperation and reduces coordination challenges by enabling direct, clear information exchange. However, improving the expressiveness of the interaction communication with MASs is not without challenges. In this sense, the interplay between self-adaptive systems and effective communication is crucial for future MAS advancements. In this paper, we propose the integration of large language models (LLMs) such as GPT-based technologies into multiagent systems. We anchor our methodology on the MAPE-K model, which is renowned for its robust support in monitoring, analyzing, planning, and executing system adaptations in response to dynamic environments. We also present a practical illustration of the proposed approach, in which we implement and assess a basic MAS-based application. The approach significantly advances the state-of-the-art of self-adaptive systems by proposing a new paradigm for MAS self-adaptation of autonomous systems based on LLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.02309v2">CodeGen2: Lessons for Training LLMs on Programming and Natural Languages</a></div>
    <div class="paper-meta">
      📅 2023-07-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable abilities in representation learning for program synthesis and understanding tasks. The quality of the learned representations appears to be dictated by the neural scaling laws as a function of the number of model parameters and observations, while imposing upper bounds on the model performance by the amount of available data and compute, which is costly. In this study, we attempt to render the training of LLMs for program synthesis more efficient by unifying four key components: (1) model architectures, (2) learning methods, (3) infill sampling, and, (4) data distributions. Specifically, for the model architecture, we attempt to unify encoder and decoder-based models into a single prefix-LM. For learning methods, (i) causal language modeling, (ii) span corruption, (iii) infilling are unified into a simple learning algorithm. For infill sampling, we explore the claim of a "free lunch" hypothesis. For data distributions, the effect of a mixture distribution and multi-epoch training of programming and natural languages on model performance is explored. We conduct a comprehensive series of empirical experiments on 1B LLMs, for which failures and successes of this exploration are distilled into five lessons. We will provide a final recipe for training and release CodeGen2 models in size 1B, 3.7B, 7B, and, 16B parameters, along with the training framework as open-source: https://github.com/salesforce/CodeGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.05337v1">Explaining Competitive-Level Programming Solutions using LLMs</a></div>
    <div class="paper-meta">
      📅 2023-07-11
      | 💬 14 pages, presented at the 1st NLRSE workshop
    </div>
    <details class="paper-abstract">
      In this paper, we approach competitive-level programming problem-solving as a composite task of reasoning and code generation. We propose a novel method to automatically annotate natural language explanations to \textit{<problem, solution>} pairs. We show that despite poor performance in solving competitive-level programming problems, state-of-the-art LLMs exhibit a strong capacity in describing and explaining solutions. Our explanation generation methodology can generate a structured solution explanation for the problem containing descriptions and analysis. To evaluate the quality of the annotated explanations, we examine their effectiveness in two aspects: 1) satisfying the human programming expert who authored the oracle solution, and 2) aiding LLMs in solving problems more effectively. The experimental results on the CodeContests dataset demonstrate that while LLM GPT3.5's and GPT-4's abilities in describing the solution are comparable, GPT-4 shows a better understanding of the key idea behind the solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.07411v1">Detecting LLM-Generated Text in Computing Education: A Comparative Study for ChatGPT Cases</a></div>
    <div class="paper-meta">
      📅 2023-07-10
      | 💬 18 pages total (16 pages, 2 reference pages). In submission
    </div>
    <details class="paper-abstract">
      Due to the recent improvements and wide availability of Large Language Models (LLMs), they have posed a serious threat to academic integrity in education. Modern LLM-generated text detectors attempt to combat the problem by offering educators with services to assess whether some text is LLM-generated. In this work, we have collected 124 submissions from computer science students before the creation of ChatGPT. We then generated 40 ChatGPT submissions. We used this data to evaluate eight publicly-available LLM-generated text detectors through the measures of accuracy, false positives, and resilience. The purpose of this work is to inform the community of what LLM-generated text detectors work and which do not, but also to provide insights for educators to better maintain academic integrity in their courses. Our results find that CopyLeaks is the most accurate LLM-generated text detector, GPTKit is the best LLM-generated text detector to reduce false positives, and GLTR is the most resilient LLM-generated text detector. We also express concerns over 52 false positives (of 114 human written submissions) generated by GPTZero. Finally, we note that all LLM-generated text detectors are less accurate with code, other languages (aside from English), and after the use of paraphrasing tools (like QuillBot). Modern detectors are still in need of improvements so that they can offer a full-proof solution to help maintain academic integrity. Further, their usability can be improved by facilitating a smooth API integration, providing clear documentation of their features and the understandability of their model(s), and supporting more commonly used languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.04492v1">Calculating Originality of LLM Assisted Source Code</a></div>
    <div class="paper-meta">
      📅 2023-07-10
    </div>
    <details class="paper-abstract">
      The ease of using a Large Language Model (LLM) to answer a wide variety of queries and their high availability has resulted in LLMs getting integrated into various applications. LLM-based recommenders are now routinely used by students as well as professional software programmers for code generation and testing. Though LLM-based technology has proven useful, its unethical and unattributed use by students and professionals is a growing cause of concern. As such, there is a need for tools and technologies which may assist teachers and other evaluators in identifying whether any portion of a source code is LLM generated. In this paper, we propose a neural network-based tool that instructors can use to determine the original effort (and LLM's contribution) put by students in writing source codes. Our tool is motivated by minimum description length measures like Kolmogorov complexity. Our initial experiments with moderate sized (up to 500 lines of code) have shown promising results that we report in this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.07422v1">Can LLMs be Good Financial Advisors?: An Initial Study in Personal Decision Making for Optimized Outcomes</a></div>
    <div class="paper-meta">
      📅 2023-07-08
    </div>
    <details class="paper-abstract">
      Increasingly powerful Large Language Model (LLM) based chatbots, like ChatGPT and Bard, are becoming available to users that have the potential to revolutionize the quality of decision-making achieved by the public. In this context, we set out to investigate how such systems perform in the personal finance domain, where financial inclusion has been an overarching stated aim of banks for decades. We asked 13 questions representing banking products in personal finance: bank account, credit card, and certificate of deposits and their inter-product interactions, and decisions related to high-value purchases, payment of bank dues, and investment advice, and in different dialects and languages (English, African American Vernacular English, and Telugu). We find that although the outputs of the chatbots are fluent and plausible, there are still critical gaps in providing accurate and reliable financial information using LLM-based chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.02839v2">Enhancing LLM with Evolutionary Fine Tuning for News Summary Generation</a></div>
    <div class="paper-meta">
      📅 2023-07-08
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      News summary generation is an important task in the field of intelligence analysis, which can provide accurate and comprehensive information to help people better understand and respond to complex real-world events. However, traditional news summary generation methods face some challenges, which are limited by the model itself and the amount of training data, as well as the influence of text noise, making it difficult to generate reliable information accurately. In this paper, we propose a new paradigm for news summary generation using LLM with powerful natural language understanding and generative capabilities. We use LLM to extract multiple structured event patterns from the events contained in news paragraphs, evolve the event pattern population with genetic algorithm, and select the most adaptive event pattern to input into the LLM to generate news summaries. A News Summary Generator (NSG) is designed to select and evolve the event pattern populations and generate news summaries. The experimental results show that the news summary generator is able to generate accurate and reliable news summaries with some generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.06767v2">The Impact of ChatGPT and LLMs on Medical Imaging Stakeholders: Perspectives and Use Cases</a></div>
    <div class="paper-meta">
      📅 2023-07-06
      | 💬 Paper invited for the first issue of Meta-Radiology
    </div>
    <details class="paper-abstract">
      This study investigates the transformative potential of Large Language Models (LLMs), such as OpenAI ChatGPT, in medical imaging. With the aid of public data, these models, which possess remarkable language understanding and generation capabilities, are augmenting the interpretive skills of radiologists, enhancing patient-physician communication, and streamlining clinical workflows. The paper introduces an analytic framework for presenting the complex interactions between LLMs and the broader ecosystem of medical imaging stakeholders, including businesses, insurance entities, governments, research institutions, and hospitals (nicknamed BIGR-H). Through detailed analyses, illustrative use cases, and discussions on the broader implications and future directions, this perspective seeks to raise discussion in strategic planning and decision-making in the era of AI-enabled healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.02628v1">SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2023-07-05
    </div>
    <details class="paper-abstract">
      Autoregressive large language models (LLMs) have made remarkable progress in various natural language generation tasks. However, they incur high computation cost and latency resulting from the autoregressive token-by-token generation. To address this issue, several approaches have been proposed to reduce computational cost using early-exit strategies. These strategies enable faster text generation using reduced computation without applying the full computation graph to each token. While existing token-level early exit methods show promising results for online inference, they cannot be readily applied for batch inferencing and Key-Value caching. This is because they have to wait until the last token in a batch exits before they can stop computing. This severely limits the practical application of such techniques. In this paper, we propose a simple and effective token-level early exit method, SkipDecode, designed to work seamlessly with batch inferencing and KV caching. It overcomes prior constraints by setting up a singular exit point for every token in a batch at each sequence position. It also guarantees a monotonic decrease in exit points, thereby eliminating the need to recompute KV Caches for preceding tokens. Rather than terminating computation prematurely as in prior works, our approach bypasses lower to middle layers, devoting most of the computational resources to upper layers, allowing later tokens to benefit from the compute expenditure by earlier tokens. Our experimental results show that SkipDecode can obtain 2x to 5x inference speedups with negligible regression across a variety of tasks. This is achieved using OPT models of 1.3 billion and 6.7 billion parameters, all the while being directly compatible with batching and KV caching optimization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.10188v1">Several categories of Large Language Models (LLMs): A Short Survey</a></div>
    <div class="paper-meta">
      📅 2023-07-05
    </div>
    <details class="paper-abstract">
      Large Language Models(LLMs)have become effective tools for natural language processing and have been used in many different fields. This essay offers a succinct summary of various LLM subcategories. The survey emphasizes recent developments and efforts made for various LLM kinds, including task-based financial LLMs, multilingual language LLMs, biomedical and clinical LLMs, vision language LLMs, and code language models. The survey gives a general summary of the methods, attributes, datasets, transformer models, and comparison metrics applied in each category of LLMs. Furthermore, it highlights unresolved problems in the field of developing chatbots and virtual assistants, such as boosting natural language processing, enhancing chatbot intelligence, and resolving moral and legal dilemmas. The purpose of this study is to provide readers, developers, academics, and users interested in LLM-based chatbots and virtual intelligent assistant technologies with useful information and future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.15195v2">Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic</a></div>
    <div class="paper-meta">
      📅 2023-07-03
    </div>
    <details class="paper-abstract">
      In human conversations, individuals can indicate relevant regions within a scene while addressing others. In turn, the other person can then respond by referring to specific regions if necessary. This natural referential ability in dialogue remains absent in current Multimodal Large Language Models (MLLMs). To fill this gap, this paper proposes an MLLM called Shikra, which can handle spatial coordinate inputs and outputs in natural language. Its architecture consists of a vision encoder, an alignment layer, and a LLM. It is designed to be straightforward and simple, without the need for extra vocabularies, position encoder, pre-/post-detection modules, or external plug-in models. All inputs and outputs are in natural language form. Referential dialogue is a superset of various vision-language (VL) tasks. Shikra can naturally handle location-related tasks like REC and PointQA, as well as conventional VL tasks such as Image Captioning and VQA. Experimental results showcase Shikra's promising performance. Furthermore, it enables numerous exciting applications, like providing mentioned objects' coordinates in chains of thoughts and comparing user-pointed regions similarities. Our code, model and dataset are accessed at https://github.com/shikras/shikra.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.01128v1">Iterative Zero-Shot LLM Prompting for Knowledge Graph Construction</a></div>
    <div class="paper-meta">
      📅 2023-07-03
    </div>
    <details class="paper-abstract">
      In the current digitalization era, capturing and effectively representing knowledge is crucial in most real-world scenarios. In this context, knowledge graphs represent a potent tool for retrieving and organizing a vast amount of information in a properly interconnected and interpretable structure. However, their generation is still challenging and often requires considerable human effort and domain expertise, hampering the scalability and flexibility across different application fields. This paper proposes an innovative knowledge graph generation approach that leverages the potential of the latest generative large language models, such as GPT-3.5, that can address all the main critical issues in knowledge graph building. The approach is conveyed in a pipeline that comprises novel iterative zero-shot and external knowledge-agnostic strategies in the main stages of the generation process. Our unique manifold approach may encompass significant benefits to the scientific community. In particular, the main contribution can be summarized by: (i) an innovative strategy for iteratively prompting large language models to extract relevant components of the final graph; (ii) a zero-shot strategy for each prompt, meaning that there is no need for providing examples for "guiding" the prompt result; (iii) a scalable solution, as the adoption of LLMs avoids the need for any external resources or human expertise. To assess the effectiveness of our proposed model, we performed experiments on a dataset that covered a specific domain. We claim that our proposal is a suitable solution for scalable and versatile knowledge graph construction and may be applied to different and novel contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.00461v1">Conformer LLMs -- Convolution Augmented Large Language Models</a></div>
    <div class="paper-meta">
      📅 2023-07-02
      | 💬 6 pages, 1 figure
    </div>
    <details class="paper-abstract">
      This work builds together two popular blocks of neural architecture, namely convolutional layers and Transformers, for large language models (LLMs). Non-causal conformers are used ubiquitously in automatic speech recognition. This work aims to adapt these architectures in a causal setup for training LLMs. Transformers decoders effectively capture long-range dependencies over several modalities and form a core backbone of modern advancements in machine learning. Convolutional architectures have been popular in extracting features in domains such as raw 1-D signals, speech, and images, to name a few. In this paper, by combining local and global dependencies over latent representations using causal convolutional filters and Transformer, we achieve significant gains in performance. This work showcases a robust speech architecture that can be integrated and adapted in a causal setup beyond speech applications for large-scale language modeling.
    </details>
</div>
