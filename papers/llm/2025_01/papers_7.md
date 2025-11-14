# llm - 2025_01

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00982v1">Are LLMs effective psychological assessors? Leveraging adaptive RAG for interpretable mental health screening through psychometric practice</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      In psychological practice, standardized questionnaires serve as essential tools for assessing mental constructs (e.g., attitudes, traits, and emotions) through structured questions (aka items). With the increasing prevalence of social media platforms where users share personal experiences and emotions, researchers are exploring computational methods to leverage this data for rapid mental health screening. In this study, we propose a novel adaptive Retrieval-Augmented Generation (RAG) approach that completes psychological questionnaires by analyzing social media posts. Our method retrieves the most relevant user posts for each question in a psychological survey and uses Large Language Models (LLMs) to predict questionnaire scores in a zero-shot setting. Our findings are twofold. First we demonstrate that this approach can effectively predict users' responses to psychological questionnaires, such as the Beck Depression Inventory II (BDI-II), achieving performance comparable to or surpassing state-of-the-art models on Reddit-based benchmark datasets without relying on training data. Second, we show how this methodology can be generalized as a scalable screening tool, as the final assessment is systematically derived by completing standardized questionnaires and tracking how individual item responses contribute to the diagnosis, aligning with established psychometric practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13184v5">What if LLMs Have Different World Views: Simulating Alien Civilizations with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-01
    </div>
    <details class="paper-abstract">
      This study introduces "CosmoAgent," an innovative artificial intelligence system that utilizes Large Language Models (LLMs) to simulate complex interactions between human and extraterrestrial civilizations. This paper introduces a mathematical model for quantifying the levels of civilization development and further employs a state transition matrix approach to evaluate their trajectories. Through this methodology, our study quantitatively analyzes the growth trajectories of civilizations, providing insights into future decision-making at critical points of growth and saturation. Furthermore, this paper acknowledges the vast diversity of potential living conditions across the universe, which could foster unique cosmologies, ethical codes, and worldviews among different civilizations. Recognizing the Earth-centric bias inherent in current LLM designs, we propose the novel concept of using LLM agents with diverse ethical paradigms and simulating interactions between entities with distinct moral principles. This innovative research not only introduces a novel method for comprehending potential inter-civilizational dynamics but also holds practical value in enabling entities with divergent value systems to strategize, prevent conflicts, and engage in games under conditions of asymmetric information. The accompanying code is available at https://github.com/MingyuJ666/Simulating-Alien-Civilizations-with-LLM-based-Agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00911v1">Aligning LLMs with Domain Invariant Reward Models</a></div>
    <div class="paper-meta">
      📅 2025-01-01
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) to human preferences is challenging in domains where preference data is unavailable. We address the problem of learning reward models for such target domains by leveraging feedback collected from simpler source domains, where human preferences are easier to obtain. Our key insight is that, while domains may differ significantly, human preferences convey \emph{domain-agnostic} concepts that can be effectively captured by a reward model. We propose \method, a framework that trains domain-invariant reward models by optimizing a dual loss: a domain loss that minimizes the divergence between source and target distribution, and a source loss that optimizes preferences on the source domain. We show \method is a general approach that we evaluate and analyze across 4 distinct settings: (1) Cross-lingual transfer (accuracy: $0.621 \rightarrow 0.661$), (2) Clean-to-noisy (accuracy: $0.671 \rightarrow 0.703$), (3) Few-shot-to-full transfer (accuracy: $0.845 \rightarrow 0.920$), and (4) Simple-to-complex tasks transfer (correlation: $0.508 \rightarrow 0.556$). Our code, models and data are available at \url{https://github.com/portal-cornell/dial}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00829v1">An LLM-Empowered Adaptive Evolutionary Algorithm For Multi-Component Deep Learning Systems</a></div>
    <div class="paper-meta">
      📅 2025-01-01
      | 💬 9
    </div>
    <details class="paper-abstract">
      Multi-objective evolutionary algorithms (MOEAs) are widely used for searching optimal solutions in complex multi-component applications. Traditional MOEAs for multi-component deep learning (MCDL) systems face challenges in enhancing the search efficiency while maintaining the diversity. To combat these, this paper proposes $\mu$MOEA, the first LLM-empowered adaptive evolutionary search algorithm to detect safety violations in MCDL systems. Inspired by the context-understanding ability of Large Language Models (LLMs), $\mu$MOEA promotes the LLM to comprehend the optimization problem and generate an initial population tailed to evolutionary objectives. Subsequently, it employs adaptive selection and variation to iteratively produce offspring, balancing the evolutionary efficiency and diversity. During the evolutionary process, to navigate away from the local optima, $\mu$MOEA integrates the evolutionary experience back into the LLM. This utilization harnesses the LLM's quantitative reasoning prowess to generate differential seeds, breaking away from current optimal solutions. We evaluate $\mu$MOEA in finding safety violations of MCDL systems, and compare its performance with state-of-the-art MOEA methods. Experimental results show that $\mu$MOEA can significantly improve the efficiency and diversity of the evolutionary search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00805v1">SLIDE: Integrating Speech Language Model with LLM for Spontaneous Spoken Dialogue Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-01
      | 💬 Accepted by ICASSP 2025
    </div>
    <details class="paper-abstract">
      Recently, ``textless" speech language models (SLMs) based on speech units have made huge progress in generating naturalistic speech, including non-verbal vocalizations. However, the generated speech samples often lack semantic coherence. In this paper, we propose SLM and LLM Integration for spontaneous spoken Dialogue gEneration (SLIDE). Specifically, we first utilize an LLM to generate the textual content of spoken dialogue. Next, we convert the textual dialogues into phoneme sequences and use a two-tower transformer-based duration predictor to predict the duration of each phoneme. Finally, an SLM conditioned on the spoken phoneme sequences is used to vocalize the textual dialogue. Experimental results on the Fisher dataset demonstrate that our system can generate naturalistic spoken dialogue while maintaining high semantic coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03104v2">ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-01-01
    </div>
    <details class="paper-abstract">
      Understanding time series is crucial for its application in real-world scenarios. Recently, large language models (LLMs) have been increasingly applied to time series tasks, leveraging their strong language capabilities to enhance various applications. However, research on multimodal LLMs (MLLMs) for time series understanding and reasoning remains limited, primarily due to the scarcity of high-quality datasets that align time series with textual information. This paper introduces ChatTS, a novel MLLM designed for time series analysis. ChatTS treats time series as a modality, similar to how vision MLLMs process images, enabling it to perform both understanding and reasoning with time series. To address the scarcity of training data, we propose an attribute-based method for generating synthetic time series with detailed attribute descriptions. We further introduce Time Series Evol-Instruct, a novel approach that generates diverse time series Q&As, enhancing the model's reasoning capabilities. To the best of our knowledge, ChatTS is the first TS-MLLM that takes multivariate time series as input for understanding and reasoning, which is fine-tuned exclusively on synthetic datasets. We evaluate its performance using benchmark datasets with real-world data, including six alignment tasks and four reasoning tasks. Our results show that ChatTS significantly outperforms existing vision-based MLLMs (e.g., GPT-4o) and text/agent-based LLMs, achieving a 46.0% improvement in alignment tasks and a 25.8% improvement in reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16882v2">PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality and Mental Health</a></div>
    <div class="paper-meta">
      📅 2025-01-01
    </div>
    <details class="paper-abstract">
      Artificial intelligence-based language generators are now a part of most people's lives. However, by default, they tend to generate "average" language without reflecting the ways in which people differ. Here, we propose a lightweight modification to the standard language model transformer architecture - "PsychAdapter" - that uses empirically derived trait-language patterns to generate natural language for specified personality, demographic, and mental health characteristics (with or without prompting). We applied PsychAdapters to modify OpenAI's GPT-2, Google's Gemma, and Meta's Llama 3 and found generated text to reflect the desired traits. For example, expert raters evaluated PsychAdapter's generated text output and found it matched intended trait levels with 87.3% average accuracy for Big Five personalities, and 96.7% for depression and life satisfaction. PsychAdapter is a novel method to introduce psychological behavior patterns into language models at the foundation level, independent of prompting, by influencing every transformer layer. This approach can create chatbots with specific personality profiles, clinical training tools that mirror language associated with psychological conditionals, and machine translations that match an authors reading or education level without taking up LLM context windows. PsychAdapter also allows for the exploration psychological constructs through natural language expression, extending the natural language processing toolkit to study human psychology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00684v1">IGC: Integrating a Gated Calculator into an LLM to Solve Arithmetic Tasks Reliably and Efficiently</a></div>
    <div class="paper-meta">
      📅 2025-01-01
    </div>
    <details class="paper-abstract">
      Solving arithmetic tasks is a simple and fundamental skill, yet modern Large Language Models (LLMs) have great difficulty with them. We introduce the Integrated Gated Calculator (IGC), a module that enables LLMs to perform arithmetic by emulating a calculator on the GPU. We finetune a Llama model with our module and test it on the BigBench Arithmetic benchmark, where it beats the State of the Art, outperforming all models on the benchmark, including models almost two orders of magnitude larger. Our approach takes only a single iteration to run and requires no external tools. It performs arithmetic operations entirely inside the LLM without the need to produce intermediate tokens. It is computationally efficient, interpretable, and avoids side-effects on tasks that do not require arithmetic operations. It reliably achieves 98\% to 99\% accuracy across multiple training runs and for all subtasks, including the substantially harder subtask of multiplication, which was previously unsolved.
    </details>
</div>
