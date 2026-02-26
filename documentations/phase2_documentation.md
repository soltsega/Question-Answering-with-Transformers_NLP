# Phase II: Data Acquisition & EDA - Complete Documentation

## Overview
This phase focused on loading the SQuAD v1.1 dataset and performing comprehensive Exploratory Data Analysis (EDA) to understand the data characteristics that inform model choices and preprocessing decisions.

## Prerequisites & Environment Setup

### Required Dependencies
```bash
pip install transformers datasets torch pandas numpy plotly
```

### Compatibility Issues & Solutions
**Issue**: NumPy 2.x compatibility problems with matplotlib
**Solution**: Use one of these approaches:
1. **Plotly (Recommended)**: `pip install plotly` - No compilation issues
2. **Downgrade NumPy**: `pip install "numpy<2"` 
3. **Text-based visualization**: Built-in fallback method

### Virtual Environment Setup
```bash
# Create and activate virtual environment
python -m venv localenv
# Windows
localenv\Scripts\Activate.ps1
# Linux/Mac  
source localenv/bin/activate
```

## Step-by-Step Implementation

### 1. Dataset Loading
```python
from datasets import load_dataset

# Load SQuAD v1.1 dataset
dataset = load_dataset("squad")
print(dataset)
```

**Output Structure**:
- Train: 87,599 records
- Validation: 10,570 records
- Features: ['id', 'title', 'context', 'question', 'answers']

### 2. Dataset Overview Analysis
```python
splits = ['train', 'validation']
summary = []

for split in splits:
    df = dataset[split].to_pandas()
    summary.append({
        'Split': split,
        'Total Records': len(df),
        'Unique Contexts': df['context'].nunique(),
        'Unique Titles': df['title'].nunique()
    })

pd.DataFrame(summary)
```

**Key Findings**:
- Train: 87,599 records, 18,891 unique contexts, 442 unique titles
- Validation: 10,570 records, 2,067 unique contexts, 48 unique titles

### 3. Length Analysis
```python
train_df = dataset['train'].to_pandas()

# Calculate lengths in words (token approximation)
train_df['context_len'] = train_df['context'].apply(lambda x: len(x.split()))
train_df['question_len'] = train_df['question'].apply(lambda x: len(x.split()))
train_df['answer_len'] = train_df['answers'].apply(lambda x: len(x['text'][0].split()))

# Statistical summary
train_df[['context_len', 'question_len', 'answer_len']].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
```

**Critical Statistics for Tokenization**:
- **Context Length**: Mean=119.8, Median=110, 95th percentile=213, Max=653 words
- **Question Length**: Mean=10.1, Median=10, 95th percentile=17, Max=40 words  
- **Answer Length**: Mean=3.2, Median=2, 95th percentile=10, Max=43 words

### 4. Visualization Methods

#### Method 1: Plotly (Recommended)
```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, subplot_titles=('Context Length', 'Question Length', 'Answer Length'))

fig.add_trace(go.Histogram(x=train_df['context_len'], name='Context'), row=1, col=1)
fig.add_trace(go.Histogram(x=train_df['question_len'], name='Question'), row=1, col=2)
fig.add_trace(go.Histogram(x=train_df['answer_len'], name='Answer'), row=1, col=3)

fig.update_layout(height=400, showlegend=False)
fig.show()
```

#### Method 2: Matplotlib/Seaborn (if compatible)
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(train_df['context_len'], bins=50, ax=axes[0], color='skyblue')
axes[0].set_title('Context Length Distribution (Words)')

sns.histplot(train_df['question_len'], bins=30, ax=axes[1], color='salmon')
axes[1].set_title('Question Length Distribution (Words)')

sns.histplot(train_df['answer_len'], bins=20, ax=axes[2], color='lightgreen')
axes[2].set_title('Answer Length Distribution (Words)')

plt.tight_layout()
plt.show()
```

#### Method 3: Text-based Fallback
```python
def simple_histogram(data, bins, title):
    min_val, max_val = data.min(), data.max()
    bin_width = (max_val - min_val) / bins
    bin_counts = [0] * bins
    
    for val in data:
        bin_idx = min(int((val - min_val) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1
    
    print(f"\n{title}:")
    for i, count in enumerate(bin_counts):
        bar = '█' * (count // max(bin_counts) * 50)
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        print(f"{bin_start:5.0f}-{bin_end:5.0f}: {bar} ({count})")
```

### 5. Topic Analysis
```python
print("Top 15 Topics in SQuAD v1.1:")
top_topics = train_df['title'].value_counts()[:15]
for i, (title, count) in enumerate(top_topics.items(), 1):
    print(f"{i:2d}. {title}: {count}")
```

**Top Topics**: Wikipedia articles covering various domains (history, science, geography, etc.)

### 6. Data Quality Analysis
```python
from collections import Counter

# Answer position analysis
answer_positions = []
for item in dataset['train']:
    if item['answers']['answer_start']:
        answer_positions.append(item['answers']['answer_start'][0])

print(f"Mean answer start position: {np.mean(answer_positions):.1f} characters")
print(f"Median answer start position: {np.median(answer_positions):.1f} characters")

# Question starting words
question_start_words = [q.split()[0].lower() for q in train_df['question'] if q.split()]
start_word_counts = Counter(question_start_words)

print("\nTop 10 question starting words:")
for word, count in start_word_counts.most_common(10):
    print(f"  {word}: {count}")

# Data quality checks
empty_answers = train_df['answer_len'].apply(lambda x: x == 0).sum()
empty_questions = train_df['question_len'].apply(lambda x: x == 0).sum()

print(f"\nEmpty answers: {empty_answers}")
print(f"Empty questions: {empty_questions}")
```

**Data Quality Results**:
- Mean answer start position: ~600-800 characters into context
- Top question starters: "What", "How", "Why", "When", "Where", "Who"
- No empty answers or questions found

### 7. Sample Data Inspection
```python
# Long context example
long_sample = train_df.sort_values(by='context_len', ascending=False).iloc[0]
print(f"Long Context - Title: {long_sample['title']}")
print(f"Length: {long_sample['context_len']} words")
print(f"Question: {long_sample['question']}")
print(f"Answer: {long_sample['answers']['text'][0]}")

# Short context example  
short_sample = train_df.sort_values(by='context_len', ascending=True).iloc[0]
print(f"\nShort Context - Title: {short_sample['title']}")
print(f"Length: {short_sample['context_len']} words")
print(f"Question: {short_sample['question']}")
print(f"Answer: {short_sample['answers']['text'][0]}")
```

### 8. Save Results
```python
import json
import os

# Save sample record
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

sample_record = dataset['train'][0]
with open(os.path.join(data_dir, "squad_sample.json"), "w") as f:
    json.dump(sample_record, f, indent=4)

print(f"Sample saved to {os.path.join(data_dir, 'squad_sample.json')}")
```

## Key Insights & Recommendations

### For Tokenization Strategy
1. **Max Sequence Length**: Set to 384 tokens (covers 95% of contexts + questions + answers)
2. **Sliding Window**: Implement for contexts > 384 tokens (5% of cases)
3. **Doc Stride**: Use 128 tokens to maintain context continuity

### For Model Architecture
1. **DistilBERT**: Suitable given moderate context lengths
2. **Answer Span Prediction**: Most answers are short (2-3 words)
3. **No Unanswerable Questions**: SQuAD v1.1 has all questions answerable

### For Training Strategy
1. **Batch Size**: 16-32 samples given moderate sequence lengths
2. **Learning Rate**: 2e-5 to 5e-5 (standard for transformer fine-tuning)
3. **Epochs**: 2-3 epochs typically sufficient for SQuAD

## Troubleshooting Guide

### Common Issues & Solutions

1. **NumPy/Matplotlib Compatibility**
   - **Problem**: `numpy.core.multiarray failed to import`
   - **Solution**: Use Plotly or downgrade NumPy: `pip install "numpy<2"`

2. **Memory Issues with Full Dataset**
   - **Problem**: Out of memory with 87k training samples
   - **Solution**: Use subset: `load_dataset("squad", split="train[:5000]")`

3. **Visualization Not Displaying**
   - **Problem**: Plots not showing in notebook
   - **Solution**: Use Plotly (interactive) or restart notebook kernel

4. **Slow Dataset Loading**
   - **Problem**: Long download times for SQuAD
   - **Solution**: Set HF_TOKEN environment variable for faster downloads

## File Structure
```
Question Answering with Transformers_NLP/
├── data/
│   ├── squad_sample.json
│   └── visualizations/
│       └── length_distributions.png
├── docs/
│   └── phase2_documentation.md
├── 01_data_exploration.ipynb
├── requirements.txt
└── localenv/
```

## Challenges Faced & Limitations

### Major Technical Challenges

#### 1. NumPy/Matplotlib Compatibility Crisis
**Problem**: 
- Matplotlib 3.7.2 was compiled with NumPy 1.x but system had NumPy 2.4.2
- Error: `numpy.core.multiarray failed to import` and `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.2`
- This completely blocked all visualization attempts

**Root Cause**: 
- NumPy 2.0 introduced breaking changes in the C API
- Pre-compiled matplotlib wheels were built against NumPy 1.x
- Virtual environment conflicts between system Python and venv installations

**Solutions Attempted**:
1. **Downgrade NumPy**: `pip install "numpy<2"` - Failed due to cached compiled modules
2. **Reinstall matplotlib**: `pip uninstall matplotlib seaborn -y && pip install matplotlib==3.7.2` - Still failed
3. **Use older versions**: `matplotlib==3.5.3, seaborn==0.11.2` - Failed due to missing Visual C++ build tools
4. **Pre-compiled wheels only**: `pip install --only-binary=all` - Failed
5. **Final Solution**: Switch to Plotly which has no NumPy compilation dependencies

**Lesson**: Always check NumPy version compatibility before installing scientific computing packages

#### 2. Virtual Environment Complexity
**Problem**: 
- Mixed installations between system Python and virtual environment
- Packages installed in system Python were not available in venv
- Conflicting package versions across environments

**Solution**: Used system-level matplotlib/seaborn installations as fallback

**Lesson**: Consistent environment management is critical for reproducible ML workflows

#### 3. Windows-Specific Build Issues
**Problem**: 
- `error: Microsoft Visual C++ 14.0 or greater is required` when building from source
- Windows lacks built-in compilers unlike Linux/macOS

**Impact**: Prevented installation of older matplotlib versions that might have been compatible

**Workaround**: Used pre-compiled packages or alternative libraries

### Data Analysis Limitations

#### 1. Token Approximation Inaccuracy
**Limitation**: Used word count (`len(x.split())`) as token approximation
**Reality**: BERT/DistilBERT tokenization uses subword units, actual token counts differ by 20-30%
**Impact**: Max sequence length recommendations may need adjustment

**Mitigation**: Phase III will use actual tokenizer for precise measurements

#### 2. Sample Size vs Computational Resources
**Trade-off**: Full dataset (87,599 samples) vs manageable subset for analysis
**Approach**: Used full dataset for statistics, but visualizations could benefit from sampling
**Impact**: Some visualizations might be memory-intensive on lower-spec machines

#### 3. Context Length Distribution Skew
**Finding**: Highly right-skewed distribution (mean=119, max=653)
**Challenge**: Determining optimal max_length that balances coverage vs computational efficiency
**Current Solution**: 95th percentile (213 words) as baseline, will refine in Phase III

### Tool & Infrastructure Limitations

#### 1. Jupyter Notebook Cell Execution Order
**Problem**: Variables not defined when cells executed out of order
**Example**: `NameError: name 'train_df' is not defined`
**Impact**: Required strict sequential execution and careful state management

**Mitigation**: Added clear cell dependencies and error handling

#### 2. Visualization Library Dependencies
**Problem**: Heavy dependency chains (matplotlib → numpy → compiled C extensions)
**Impact**: Single point of failure for entire visualization pipeline

**Alternative Adopted**: Plotly with lighter dependency footprint

#### 3. Dataset Download Reliability
**Issue**: Hugging Face dataset downloads can be slow without authentication
**Warning**: `You are sending unauthenticated requests to the HF Hub`
**Impact**: Rate limiting and slower initial setup

**Recommendation**: Set HF_TOKEN environment variable for production use

### Analysis Scope Limitations

#### 1. Language & Domain Specificity
**Scope**: Only English Wikipedia-based SQuAD v1.1 data
**Limitation**: Findings may not generalize to:
- Other languages
- Medical/legal domains
- Conversational QA
- Multi-hop reasoning

#### 2. Answer Quality Assessment
**Missing**: Analysis of answer correctness or ambiguity
**Focus**: Length and position analysis only
**Future Need**: Error analysis in Phase V

#### 3. Context Complexity Analysis
**Gap**: No analysis of:
- Syntactic complexity
- Named entity density  
- Semantic difficulty
- Reasoning requirements

### Performance & Scalability Concerns

#### 1. Memory Usage
**Issue**: Loading full dataset into pandas DataFrame
**Memory**: ~87,599 × multiple text fields = significant RAM usage
**Risk**: May not scale to larger datasets

#### 2. Computation Time
**Bottleneck**: Answer position analysis requires iterating through all samples
**Time**: Several seconds for position analysis on full dataset
**Scaling**: Would be problematic for datasets 10x larger

### Reproducibility Challenges

#### 1. Environment Drift
**Risk**: Package versions updated over time
**Mitigation**: Fixed version numbers in requirements.txt
**Remaining Risk**: System-level dependencies may change

#### 2. Data Versioning
**Issue**: SQuAD dataset version not explicitly pinned
**Risk**: Future updates might change dataset characteristics
**Solution**: Consider dataset version locking in production

#### 3. Random Seed Management
**Gap**: No random seed set for any operations
**Impact**: Any stochastic processes not reproducible
**Fix**: Add `np.random.seed(42)` and similar for future phases

### Lessons Learned for Future Phases

#### 1. Environment First Approach
- Always verify package compatibility before starting
- Use dependency matrix testing for complex environments
- Have backup visualization libraries ready

#### 2. Incremental Development
- Test imports before building complex analysis
- Use smaller subsets for development, full dataset for final analysis
- Implement graceful degradation (text-based fallbacks)

#### 3. Documentation During Development
- Record errors and solutions as they occur
- Document rationale for technical decisions
- Maintain troubleshooting log for future reference

#### 4. Cross-Platform Considerations
- Test on multiple operating systems if possible
- Avoid platform-specific dependencies when alternatives exist
- Provide Windows-specific installation instructions

## Next Phase Preparation
This EDA provides the foundation for Phase III (Preprocessing & Tokenization):
- Tokenization parameters determined (max_length=384, doc_stride=128)
- Data quality confirmed (no missing data, clean answers)
- Model choice informed (DistilBERT suitable for data characteristics)
- **Key Lesson**: Verify tokenizer compatibility early to avoid similar issues

## Reproduction Commands
```bash
# Complete setup and execution
git clone [repository]
cd "Question Answering with Transformers_NLP"
python -m venv localenv
localenv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook 01_data_exploration.ipynb
# Execute cells sequentially from top to bottom
# Note: If matplotlib fails, use plotly (recommended) or text-based fallbacks
```

This documentation ensures complete reproducibility of Phase II EDA analysis, including all challenges encountered and solutions implemented.
