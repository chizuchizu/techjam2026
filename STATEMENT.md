3. Implement a GPU Kernel for a Transformer Layer
In response to some queries from our Early Bird participants, our engineers have provided updates to the problem statement to improve clarity and to support participants better.
Problem Statement last updated: 27 August 2026, 6:25PM
Added Appendix: Test Shapes
Updated torch_transformer_benchmark.py
Technical Workshop Webinar with Q&A will be held on 28 Aug, 3:00 to 3:45pm.
Click here to join the webinar!
3.1 Background
Transformer is a widely used neural network architecture in modern AI. It is the core structure behind many natural language processing, computer vision, speech, recommendation, and large language model systems.
The main idea of Transformer is self-attention. Self-attention allows each token in a sequence to interact with other tokens directly. Compared with recurrent models, Transformer can process tokens in parallel, which makes it suitable for GPU acceleration.
Given an input sequence represented as a matrix:
$$X \in \mathbb{R}^{N \times d}$$
where $$N
$$ is the sequence length and $$d$$ is the hidden dimension, the Transformer first projects the input into Query, Key, and Value matrices:
$$Q = XW_Q$$
$$K = XW_K$$
$$V = XW_V$$
The scaled dot-product attention is computed as:
$$\text{Attention}(Q, K, V) =
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
where $$d_k
$$ is the dimension of each attention head. The scaling factor $$\sqrt{d_k}$$ is used to prevent the dot-product values from becoming too large, which could make the softmax distribution unstable.
However, the computation of Transformer is expensive. Important operations include matrix multiplication, attention score calculation, softmax, normalization, and feed-forward layers. These operations may be limited by GPU compute throughput, memory bandwidth, cache efficiency, kernel launch overhead, and tensor core utilization.
In this competition, participants are asked to use AI-assisted methods to optimize the runtime efficiency of a Transformer structure on a given GPU model. The optimized implementation should improve performance while keeping the output numerically correct compared with the reference implementation.
Participants may consider optimization methods such as operator fusion, memory layout optimization, reduced-precision computation, tensor core usage, softmax optimization, and custom CUDA, Triton, TensorFlow, or PyTorch implementations.
The goal of this task is to explore how AI can help developers analyze Transformer workloads, identify bottlenecks, and generate more efficient implementations for specific GPU hardware.
3.2 Problem Statement
- Given a fixed formula of transformer layer, participants need to submit one or several GPU kernels that implement the layers that can pass the given test cases.
- The test cases would be written in pytorch or tensorflow and the participants can modify the layer implementation if they need, which means they can decide which parts of the layers should be fused into 1 kernel.
- The test case would compare the differences between the implementation of participants and the original pytorch/tensorflow implementation, the diff should be small enough (relative error < 0.02, abs error < 0.002).
- The test cases would contain different shapes of input, including large/small batchsize, large/small sequence length, large/small dimensions, etc. The participants can choose different implementations for different shapes by adding shape checks in the implementation of layers. All the combinations of input shapes will be told to the participants.
- The use of AI tools is encouraged so that the participants can implement different kernels for different input shapes in limited time.
- Optimize & test your codes on your own machine. Different methods may be used to optimize the codes depending on the machine (GPU cards) you use.
- Provide a clear tech report including details on the AI skills/tools used to get bonus points.
- What participants need to do:
  - Download the benchmark scripts (choose either torch or tensorflow, one of them would be enough).
  - Implement the customized-implementation part and optimize it as fast as you can by AI or by hand.
[Image]
  - Run the script on your own machine.
  - Provide a clear tech report illustrating what the environment is (CPU, GPU, DISK, etc), what kind of optimizations you have done, and the final test results.
3.3 Constraints & Scope
Category
Constraints & Scope Details
In scope
AI-based code generation, GPU kernel fusion, profile tools usage, etc.
Out of scope
Production-ready deployment.
3.4 Available Resources / Data
You can download 1 of these, and run it on your own machine:
Torch Benchmark script
torch_transformer_benchmark.py
Tensorflow Benchmark script
tensorflow_transformer_benchmark.py
3.5 Deliverables
1. Written Project Description (via Devpost)
- Provide a clear written description of your project that includes:
  - How your solution addresses the problem statement
  - Development tools used (e.g. VSCode, Colab, Jupyter)
  - APIs used (e.g. OpenAI GPT-4o, Google Maps API)
  - Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)
  - Datasets and assets used (e.g. Google Local Reviews dataset, manually labelled data)
2. Public Code/GitHub Repository
- Submit a link to a public Code/GitHub repository containing:
  - Well-structured, commented code covering all components of your solution
  - A README file that includes:
    - Project overview
    - Setup and installation instructions
    - Steps to reproduce your results
    - A brief reflection on your solution's limitations and what you would improve given more time
    - Team member contributions (if applicable)
3. Demo Video
Submit a short video that:
- Demonstrates your solution working end-to-end (e.g. inference results, dashboard, model predictions)
- Is uploaded to YouTube and set to public visibility
- Is linked in your Devpost description
- Does not include third-party trademarks or copyrighted content without permission
Note for backend/NLP tracks: If a front-end interface is not applicable to your solution, a walkthrough video showing API usage, inference examples, or result analysis is accepted.
3.6 Judging Criteria
Judging Criteria
Definition
Weight
Technical Execution
The solution demonstrates strong engineering fundamentals, such as well-structured code, thoughtful architecture, and effective use of APIs or models. The demo runs reliably, and the technical complexity reflects deliberate, capable decision-making.
35%
Innovation & Problem Insight
The project demonstrates originality in both idea and approach. It stands out for the sharpness of its problem understanding — how clearly the team has framed the challenge, why it matters, and how directly the solution addresses it.
20%
Impact & Relevance
The project has clear potential to deliver value to real users or stakeholders — with meaningful reach, tangible benefit, and relevance that goes beyond solving for the hackathon prompt alone.
20%
Feasibility & Practicality
The solution is realistic and buildable beyond a prototype. The approach is technically and operationally sustainable — resource usage is proportionate, the architecture holds under real-world conditions, and the implementation is grounded rather than speculative.
15%
Presentation & Communication
[Final Event Only]: The team communicates their work with clarity. The pitch tells a coherent story; from problem to solution to potential, and the team is able to respond to questions with depth, demonstrating genuine understanding of their own project.
10%
3.7 Appendix
Test shapes:
This content is only supported in a Feishu Docs
