# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00657v1">Generative Exaggeration in LLM Social Agents: Consistency, Bias, and Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      We investigate how Large Language Models (LLMs) behave when simulating political discourse on social media. Leveraging 21 million interactions on X during the 2024 U.S. presidential election, we construct LLM agents based on 1,186 real users, prompting them to reply to politically salient tweets under controlled conditions. Agents are initialized either with minimal ideological cues (Zero Shot) or recent tweet history (Few Shot), allowing one-to-one comparisons with human replies. We evaluate three model families (Gemini, Mistral, and DeepSeek) across linguistic style, ideological consistency, and toxicity. We find that richer contextualization improves internal consistency but also amplifies polarization, stylized signals, and harmful language. We observe an emergent distortion that we call "generation exaggeration": a systematic amplification of salient traits beyond empirical baselines. Our analysis shows that LLMs do not emulate users, they reconstruct them. Their outputs, indeed, reflect internal optimization dynamics more than observed behavior, introducing structural biases that compromise their reliability as social proxies. This challenges their use in content moderation, deliberative simulations, and policy modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00543v1">Reliable Annotations with Less Effort: Evaluating LLM-Human Collaboration in Search Clarifications</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 9 pages,5 figures
    </div>
    <details class="paper-abstract">
      Despite growing interest in using large language models (LLMs) to automate annotation, their effectiveness in complex, nuanced, and multi-dimensional labelling tasks remains relatively underexplored. This study focuses on annotation for the search clarification task, leveraging a high-quality, multi-dimensional dataset that includes five distinct fine-grained annotation subtasks. Although LLMs have shown impressive capabilities in general settings, our study reveals that even state-of-the-art models struggle to replicate human-level performance in subjective or fine-grained evaluation tasks. Through a systematic assessment, we demonstrate that LLM predictions are often inconsistent, poorly calibrated, and highly sensitive to prompt variations. To address these limitations, we propose a simple yet effective human-in-the-loop (HITL) workflow that uses confidence thresholds and inter-model disagreement to selectively involve human review. Our findings show that this lightweight intervention significantly improves annotation reliability while reducing human effort by up to 45%, offering a relatively scalable and cost-effective yet accurate path forward for deploying LLMs in real-world evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00507v1">LLM-Mesh: Enabling Elastic Sharing for Serverless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      The rise of LLMs has driven demand for private serverless deployments, characterized by moderate-scale models and infrequent requests. While existing solutions follow exclusive GPU deployment, we take a step back to explore modern platforms and find that: Emerging CPU architectures with built-in accelerators are capable of serving LLMs but remain underutilized, and both CPUs and GPUs can accommodate multiple LLMs simultaneously. We propose LLM-Mesh, a serverless inference scheme for small-to-mid-sized LLMs that enables elastic sharing across heterogeneous hardware. LLM-Mesh tackles three fundamental challenges: (1) precise, fine-grained compute resource allocation at token-level to handle fluctuating computational demands; (2) a coordinated and forward-looking memory scaling mechanism to detect out-of-memory hazards and reduce operational overhead; and (3) a dual approach that reduces resource fragmentation through proactive preemption and reactive bin-packing. Experimental results on 4 32-core CPUs and 4 A100 GPUs show that LLM-Meshimproves service capacity by 44% - 63% through sharing, while further leveraging CPUs boosts this to 91% - 159%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00439v1">Beyond Sociodemographic Prompting: Using Supervision to Align LLMs with Human Response Distributions</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      The ability to accurately predict how different population groups would answer subjective questions would have great value. In this work, we show that use of relatively simple supervision can greatly improve language model alignment with diverse population groups, as measured over three datasets spanning various topics. Beyond evaluating average performance, we also report how alignment varies across specific groups. The simplicity and generality of our approach promotes easy adoption, while our broad findings provide useful guidance for when to use or not use our approach in practice. By conducting evaluation over many LLMs and prompting strategies, along with open-sourcing our work, we provide a useful benchmark to stimulate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00432v1">Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Math reasoning has become the poster child of progress in large language models (LLMs), with new models rapidly surpassing human-level performance on benchmarks like MATH and AIME. But as math leaderboards improve week by week, it is worth asking: do these gains reflect broader problem-solving ability or just narrow overfitting? To answer this question, we evaluate over 20 open-weight reasoning-tuned models across a broad suite of tasks, including math, scientific QA, agent planning, coding, and standard instruction-following. We surprisingly find that most models that succeed in math fail to transfer their gains to other domains. To rigorously study this phenomenon, we conduct controlled experiments on Qwen3-14B models using math-only data but different tuning methods. We find that reinforcement learning (RL)-tuned models generalize well across domains, while supervised fine-tuning (SFT)-tuned models often forget general capabilities. Latent-space representation and token-space distribution shift analyses reveal that SFT induces substantial representation and output drift, while RL preserves general-domain structure. Our results suggest a need to rethink standard post-training recipes, particularly the reliance on SFT-distilled data for advancing reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00418v1">Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and High-Performance GPUs</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 To appear in Proceedings of the Practice and Experience in Advanced Research Computing (PEARC '25)
    </div>
    <details class="paper-abstract">
      This study presents a benchmarking analysis of the Qualcomm Cloud AI 100 Ultra (QAic) accelerator for large language model (LLM) inference, evaluating its energy efficiency (throughput per watt) and performance against leading NVIDIA (A100, H200) and AMD (MI300A) GPUs within the National Research Platform (NRP) ecosystem. A total of 15 open-source LLMs, ranging from 117 million to 90 billion parameters, are served using the vLLM framework. The QAic inference cards appears to be energy efficient and performs well in the energy efficiency metric in most cases. The findings offer insights into the potential of the Qualcomm Cloud AI 100 Ultra for high-performance computing (HPC) applications within the National Research Platform (NRP).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00406v1">Partnering with AI: A Pedagogical Feedback System for LLM Integration into Programming Education</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 This is an extended version of a poster paper accepted and published at ECTEL-2025
    </div>
    <details class="paper-abstract">
      Feedback is one of the most crucial components to facilitate effective learning. With the rise of large language models (LLMs) in recent years, research in programming education has increasingly focused on automated feedback generation to help teachers provide timely support to every student. However, prior studies often overlook key pedagogical principles, such as mastery and progress adaptation, that shape effective feedback strategies. This paper introduces a novel pedagogical framework for LLM-driven feedback generation derived from established feedback models and local insights from secondary school teachers. To evaluate this framework, we implemented a web-based application for Python programming with LLM-based feedback that follows the framework and conducted a mixed-method evaluation with eight secondary-school computer science teachers. Our findings suggest that teachers consider that, when aligned with the framework, LLMs can effectively support students and even outperform human teachers in certain scenarios through instant and precise feedback. However, we also found several limitations, such as its inability to adapt feedback to dynamic classroom contexts. Such a limitation highlights the need to complement LLM-generated feedback with human expertise to ensure effective student learning. This work demonstrates an effective way to use LLMs for feedback while adhering to pedagogical standards and highlights important considerations for future systems.
    </details>
</div>
