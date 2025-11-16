- what data sets are we using
- what librairies/ML systems are we investigating
- what are our main research questions
- how is our work different than existing/related work

1. Intro - provide a clear statement of the problem you are trying to address and 
its importance to the field. 
2. Related work and how your work complements or extends this related work. 
3. The approach you plan to take to accomplish your project. This section should 
provide some detail about the different steps you plan to take to accomplish your 
project. 
4. The research questions and a description of how you plan to answer these 
research questions. For example: what data do you plan to use? what metrics do 
you plan to extract from this data? what techniques do you plan to apply on your 
data? etc... 
5. A list of the milestones for the project and when you plan to have each 
milestone completed. 
6. A list of references. 
The project proposal is supposed to be approx. 3 pages in length, in 2 column 
IEEE format. These 3 pages do not include your bibliography. The project 
proposal will be graded! 


We may focus on one of two directions:
1. Taxonomy of ML app bugs and compare with regular app bugs
2. Taxonomy of ML lib bugs and compare with regular lib bugs
3. 

======

% The methods we are planning on using depend on a few things: literature research, available datasets, analyzing techniques, and research problem. Since many ML libraries are publicly available, we will be selecting the most popular ones that are being frequently used to solve real-world challenges. We will utilize the crawler techniques to collect the data. Afterwards, we will refine the data based on some parameters, such as bugs' popularity, and discussion severity. In addition, we might have to apply some natural language procession (NLP) techniques to filter out the non-English bugs. In addition, there is also sometimes ambiguity between the reported bugs. Therefore, we will adopt a few sentiment techniques SentiStrength, and Natural Language Toolkit (NLTK). It allows filtering out the inconsistent posts, while we could also adopt some tools like AR-Miner to remove uninformative issues and Jazzy Spell Checker for spelling corrections.

% The issues usually contain many other artifacts alongside bugs, such as documentation, feature description and so on. However, the project managers use different labels for different types of issues. We will use these labels to extract bugs out of other types of issues.

% The target systems are fairly popular and have a lot of reported bugs. For example, as of 18 February 2022, PyTorch has 7709 open and 16928 closed issues \footnote{\url{https://github.com/pytorch/pytorch/issues}}. Therefore, we will choose 50 closed and resolved bugs from this library. The issues in GitHub can be assigned by different reactions (thumbs up, heart, laughter etc). We will choose based on {}.

% After collecting the set of the bugs, we will manually go through each of them, read their description and discussions and check the corresponding fixing code and label the bugs to a category. We have not decided whether we will do open-coding or closed-coding. Either way, we will have at least two persons review each of the bugs and resolve the conflicts through discussion.

% Jia et al.\cite{jia2020empirical} conducted a similar study in 2020 only on TensorFlow. We will compare our results with theirs.

% We will look for publicly available datasets and try to extend them depending on the year it was initially released. However, in any case, we will extract the bugs from GitHub of the libraries that we plan to investigate.
 
======================
% FOUND IN INTRODUCTION AFTER "Moreover, researchers have shown that ML systems demonstrate significantly different characteristics in many aspects of software engineering \cite{amershi2019software}, requiring  different approaches to its requirements engineering \cite{vogelsang2019requirements,horkoff2019non}, design \cite{serban2021empirical,washizaki2020machine}, testing \cite{braiek2020testing} etc. than traditional software systems."

% Therein lies the importance of observing the defects in machine learning libraries because there is a direct correlation between ML software and corresponding ML libraries.
% \diego{This paragraph should be simplified: Just pinpoint some of the most important works that are directly related to ours and leave the details for the related work section ;).}
% Thung et al. \cite{thung2012empirical} analyzed the bug database of three ML-based systems and labeled the bugs into different categories. They also analyzed the relationships between different dimensions of bugs, such as the category, severity, cost of fixing, impact, etc. Kim et al. \cite{kim2021denchmark} developed a benchmark of bugs in machine learning systems with a focus to help with automatic debugging. In addition, some empirical studies have also been done on bugs in applications of a branch of ML known as deep learning. Y. Zhang et al. \cite{zhang2018empirical} conducted an empirical study of bugs in applications that call the APIs of TensorFlow \cite{abadi2016tensorflow} and their corresponding fixes to find the root causes of the bugs. Islam et al. \cite{islam2019comprehensive} analyzed applications using deep learning libraries, such as Caffe\cite{jia2014caffe}, Keras \cite{keras}, Theano \cite{bergstra2011theano}, and Torch \cite{collobert2002torch}. Humbatova et al. \cite{humbatova2020taxonomy} also conducted a similar study on applications using TensorFlow \cite{abadi2016tensorflow}, Keras \cite{keras} and PyTorch \cite{paszke2019pytorch}. X. Zhang et al. \cite{zhang2020towards} studied the bugs related to incorrect decisions provided by deep learning systems and characterized such bugs.
% While most studies focus on the bug classification and their characteristics in deep learning systems and applications, we believe that it is also important to analyze bugs that may also originate at the library level. 