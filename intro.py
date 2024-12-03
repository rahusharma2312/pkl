{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Cn3vH-Wu2HTb",
        "outputId": "343b1013-9a27-42f4-a7ac-957a635ccb07"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.46.2)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.5.1+cu121)\n",
            "Collecting datasets\n",
            "  Downloading datasets-3.1.0-py3-none-any.whl.metadata (20 kB)\n",
            "Requirement already satisfied: flask in /usr/local/lib/python3.10/dist-packages (3.0.3)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.16.1)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.26.2)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.26.4)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.2)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2024.9.11)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.32.3)\n",
            "Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.5)\n",
            "Requirement already satisfied: tokenizers<0.21,>=0.20 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.6)\n",
            "Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.10/dist-packages (from torch) (4.12.2)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.4.2)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.4)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2024.10.0)\n",
            "Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.10/dist-packages (from torch) (1.13.1)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy==1.13.1->torch) (1.3.0)\n",
            "Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (17.0.0)\n",
            "Collecting dill<0.3.9,>=0.3.0 (from datasets)\n",
            "  Downloading dill-0.3.8-py3-none-any.whl.metadata (10 kB)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (2.2.2)\n",
            "Collecting xxhash (from datasets)\n",
            "  Downloading xxhash-3.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)\n",
            "Collecting multiprocess<0.70.17 (from datasets)\n",
            "  Downloading multiprocess-0.70.16-py310-none-any.whl.metadata (7.2 kB)\n",
            "Collecting fsspec (from torch)\n",
            "  Downloading fsspec-2024.9.0-py3-none-any.whl.metadata (11 kB)\n",
            "Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.11.2)\n",
            "Requirement already satisfied: Werkzeug>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from flask) (3.1.3)\n",
            "Requirement already satisfied: itsdangerous>=2.1.2 in /usr/local/lib/python3.10/dist-packages (from flask) (2.2.0)\n",
            "Requirement already satisfied: click>=8.1.3 in /usr/local/lib/python3.10/dist-packages (from flask) (8.1.7)\n",
            "Requirement already satisfied: blinker>=1.6.2 in /usr/local/lib/python3.10/dist-packages (from flask) (1.9.0)\n",
            "Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (2.4.3)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (24.2.0)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.5.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.1.0)\n",
            "Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (0.2.0)\n",
            "Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.17.2)\n",
            "Requirement already satisfied: async-timeout<6.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (3.0.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.8.30)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)\n",
            "Downloading datasets-3.1.0-py3-none-any.whl (480 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m480.6/480.6 kB\u001b[0m \u001b[31m10.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading dill-0.3.8-py3-none-any.whl (116 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m116.3/116.3 kB\u001b[0m \u001b[31m8.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading fsspec-2024.9.0-py3-none-any.whl (179 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m179.3/179.3 kB\u001b[0m \u001b[31m12.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading multiprocess-0.70.16-py310-none-any.whl (134 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m134.8/134.8 kB\u001b[0m \u001b[31m8.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading xxhash-3.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m194.1/194.1 kB\u001b[0m \u001b[31m13.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: xxhash, fsspec, dill, multiprocess, datasets\n",
            "  Attempting uninstall: fsspec\n",
            "    Found existing installation: fsspec 2024.10.0\n",
            "    Uninstalling fsspec-2024.10.0:\n",
            "      Successfully uninstalled fsspec-2024.10.0\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "gcsfs 2024.10.0 requires fsspec==2024.10.0, but you have fsspec 2024.9.0 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed datasets-3.1.0 dill-0.3.8 fsspec-2024.9.0 multiprocess-0.70.16 xxhash-3.5.0\n"
          ]
        }
      ],
      "source": [
        "pip install transformers torch datasets flask\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FtdYctFE2MpK",
        "outputId": "d1092664-2894-4081-9388-a29c19353359"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Dataset generation complete! Saved to 'mental_wellbeing_chatbot_dataset.json'\n"
          ]
        }
      ],
      "source": [
        "import random\n",
        "import json\n",
        "\n",
        "# Predefined inputs and responses\n",
        "inputs = [\n",
        "    \"I am so stressed about my exams, what should I do?\",\n",
        "    \"I can’t manage my time, I keep procrastinating!\",\n",
        "    \"I’ve left everything for the last minute!\",\n",
        "    \"I feel overwhelmed with all the assignments.\",\n",
        "    \"I’m too tired to study anymore.\",\n",
        "    \"I’m feeling anxious about the upcoming presentation.\",\n",
        "    \"How do I get rid of stress before an exam?\",\n",
        "    \"I’m not feeling motivated to study.\",\n",
        "    \"How do I deal with constant distractions?\",\n",
        "    \"I feel like giving up. Everything feels hard.\"\n",
        "]\n",
        "\n",
        "responses = [\n",
        "    \"It’s okay to feel stressed before exams. Try to break your tasks into smaller chunks, take short breaks, and focus on one thing at a time.\",\n",
        "    \"Procrastination can be tough. Start by setting small, achievable goals, and give yourself a reward when you complete them.\",\n",
        "    \"It happens! Start by creating a to-do list. Prioritize tasks and try to focus on one task at a time. You’ve got this!\",\n",
        "    \"Take a deep breath! Make a plan by breaking down assignments into smaller tasks. Focus on one task at a time to avoid feeling overwhelmed.\",\n",
        "    \"It’s important to rest. Try a short power nap or go for a walk to recharge. A refreshed mind learns better!\",\n",
        "    \"It’s normal to feel anxious. Prepare well, practice in front of a mirror, and breathe deeply before presenting to calm your nerves.\",\n",
        "    \"Try deep breathing exercises, take short breaks while studying, and remember that a little stress can help you focus better.\",\n",
        "    \"Start small! Set a timer for 10-15 minutes and commit to studying for that time. You might find that momentum builds after you start.\",\n",
        "    \"Minimize distractions by turning off notifications and finding a quiet study space. Use techniques like the Pomodoro method to stay focused.\",\n",
        "    \"It’s okay to feel like that sometimes. Take a break, talk to someone you trust, and remember that challenges are part of growth. You’re not alone.\"\n",
        "]\n",
        "\n",
        "# Generate 1000 unique data points\n",
        "dataset = []\n",
        "for _ in range(1000):\n",
        "    input_text = random.choice(inputs)\n",
        "    response_text = random.choice(responses)\n",
        "    dataset.append({\"input\": input_text, \"response\": response_text})\n",
        "\n",
        "# Save dataset as a JSON file\n",
        "with open(\"mental_wellbeing_chatbot_dataset.json\", \"w\") as outfile:\n",
        "    json.dump(dataset, outfile, indent=4)\n",
        "\n",
        "print(\"Dataset generation complete! Saved to 'mental_wellbeing_chatbot_dataset.json'\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 49,
          "referenced_widgets": [
            "ca122f4897b047669638fc9dcbf7fb13",
            "834e5897c6374a49a3e15b07c13f9a65",
            "32bff537f2ef45af9913fe49521999b8",
            "5a271d59a18c4260a0824f8943f7115b",
            "02ff813c0c5b4262b79c400515bee803",
            "21966790e0b548748c799ede793edf01",
            "8f2fec4605b84cd5aa7320a865b103ea",
            "25316aa8fd5d462696b5de856ba6ff82",
            "b66b213bf23e4a968eb4653bc67ac258",
            "8ae626901eb143a380e31e536442e646",
            "2d28f0df9de1443ca21a4c2277988e33"
          ]
        },
        "id": "f7jJ3gwK2TmG",
        "outputId": "04d3315d-666e-4f00-e93c-5a8b79e655f7"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "ca122f4897b047669638fc9dcbf7fb13",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Saving the dataset (0/1 shards):   0%|          | 0/1000 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "from datasets import load_dataset\n",
        "\n",
        "# Load the dataset from the file\n",
        "dataset = load_dataset(\"json\", data_files={\"train\": \"mental_wellbeing_chatbot_dataset.json\"})\n",
        "\n",
        "# Function to preprocess data for GPT-2 model\n",
        "def preprocess_data(examples):\n",
        "    inputs = examples['input']\n",
        "    responses = examples['response']\n",
        "\n",
        "    # Tokenizing inputs and responses\n",
        "    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=50)\n",
        "    target_encodings = tokenizer(responses, truncation=True, padding=True, max_length=50)\n",
        "\n",
        "    # Adding labels for training\n",
        "    input_encodings['labels'] = target_encodings['input_ids']\n",
        "\n",
        "    return input_encodings\n",
        "\n",
        "# Apply the preprocessing\n",
        "train_dataset = dataset['train'].map(preprocess_data, batched=True)\n",
        "\n",
        "# Save the preprocessed data (Optional)\n",
        "train_dataset.save_to_disk(\"processed_chatbot_dataset\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4TATxx9g40cM"
      },
      "outputs": [],
      "source": [
        "from transformers import GPT2Tokenizer\n",
        "\n",
        "# Load GPT-2 tokenizer\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")\n",
        "\n",
        "# Add a padding token (using eos_token as padding)\n",
        "tokenizer.pad_token = tokenizer.eos_token\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 49,
          "referenced_widgets": [
            "3c80aa281c194dc4a2cd1b93af6d74d2",
            "35dd7fdf2dd944ee90a7ea055af49d3f",
            "df52b3f366ea474a897fddd6bf7afc58",
            "07a97e941f824b8a89eb06c51020732a",
            "693ad5e21ad640b6ac29e6da994a5689",
            "180e3ed4bd514214905c9965709463a5",
            "c7fed0e6c44c485ca1efabe4d098a23a",
            "d88461475dab493aa3c2c1f08ed4b95f",
            "e06bd5fae4654b36ac9451030988e9eb",
            "ffc10a5826bf4e419d8536bad3fccf0a",
            "022fef9b271446e99b078dc0cef4ec01"
          ]
        },
        "id": "NI4XA83m2mL6",
        "outputId": "05e2cf40-8b54-40e4-d442-850f31d07ac4"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "3c80aa281c194dc4a2cd1b93af6d74d2",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Saving the dataset (0/1 shards):   0%|          | 0/1000 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "from datasets import load_dataset\n",
        "from transformers import GPT2Tokenizer\n",
        "\n",
        "# Load the dataset from the file\n",
        "dataset = load_dataset(\"json\", data_files={\"train\": \"mental_wellbeing_chatbot_dataset.json\"})\n",
        "\n",
        "# Load GPT-2 tokenizer\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")\n",
        "\n",
        "# Add a padding token to the tokenizer\n",
        "tokenizer.pad_token = tokenizer.eos_token # using eos_token as pad_token\n",
        "# or\n",
        "# tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # adding a new pad token '[PAD]'\n",
        "\n",
        "# Function to preprocess data for GPT-2 model\n",
        "def preprocess_data(examples):\n",
        "    inputs = examples['input']\n",
        "    responses = examples['response']\n",
        "\n",
        "    # Tokenizing inputs and responses\n",
        "    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=50)\n",
        "    target_encodings = tokenizer(responses, truncation=True, padding=True, max_length=50)\n",
        "\n",
        "    # Adding labels for training\n",
        "    input_encodings['labels'] = target_encodings['input_ids']\n",
        "\n",
        "    return input_encodings\n",
        "\n",
        "# Apply the preprocessing\n",
        "train_dataset = dataset['train'].map(preprocess_data, batched=True)\n",
        "\n",
        "# Save the preprocessed data (Optional)\n",
        "train_dataset.save_to_disk(\"processed_chatbot_dataset\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kSki-8-16T6o"
      },
      "outputs": [],
      "source": [
        "def preprocess_data(examples):\n",
        "    inputs = examples['input']\n",
        "    responses = examples['response']\n",
        "\n",
        "    # Tokenize the inputs and responses\n",
        "    input_encodings = tokenizer(inputs, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "    target_encodings = tokenizer(responses, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "\n",
        "    # Ensure the labels are aligned with the inputs (the model expects labels to have the same length as inputs)\n",
        "    input_encodings['labels'] = target_encodings['input_ids']\n",
        "\n",
        "    # Flatten to remove batch dimension and ensure the padding works\n",
        "    return {k: v.squeeze() for k, v in input_encodings.items()}\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HrdyZAT16CiU"
      },
      "outputs": [],
      "source": [
        "def preprocess_data(examples):\n",
        "    inputs = examples['input']\n",
        "    responses = examples['response']\n",
        "\n",
        "    # Tokenize inputs and responses while ensuring padding and truncation are consistent\n",
        "    input_encodings = tokenizer(inputs, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "    target_encodings = tokenizer(responses, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "\n",
        "    # Adding labels for training (the response sequence is the target label for training)\n",
        "    input_encodings['labels'] = target_encodings['input_ids']\n",
        "\n",
        "    # Ensure both input and labels have the same length\n",
        "    return input_encodings\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "e6Sa1Eai6h0g"
      },
      "outputs": [],
      "source": [
        "def preprocess_data(examples):\n",
        "    inputs = examples['input']\n",
        "    responses = examples['response']\n",
        "\n",
        "    # Tokenize the inputs and responses with padding and truncation to a fixed length\n",
        "    input_encodings = tokenizer(inputs, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "    target_encodings = tokenizer(responses, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "\n",
        "    # Ensure the labels are aligned with the inputs (the model expects labels to have the same length as inputs)\n",
        "    # Both the inputs and labels should have the same length after padding\n",
        "    input_encodings['labels'] = target_encodings['input_ids']\n",
        "\n",
        "    # Ensure both input and label sequences have no extra batch dimensions or misalignment\n",
        "    return {k: v.squeeze() for k, v in input_encodings.items()}\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vkuJ6_xW6soJ"
      },
      "outputs": [],
      "source": [
        "def preprocess_data(examples):\n",
        "    inputs = examples['input']\n",
        "    responses = examples['response']\n",
        "\n",
        "    # Tokenize the inputs and responses with padding and truncation to a fixed length\n",
        "    input_encodings = tokenizer(\n",
        "        inputs,\n",
        "        truncation=True,\n",
        "        padding='max_length',\n",
        "        max_length=50,\n",
        "        return_tensors=\"pt\"\n",
        "    )\n",
        "    target_encodings = tokenizer(\n",
        "        responses,\n",
        "        truncation=True,\n",
        "        padding='max_length',\n",
        "        max_length=50,\n",
        "        return_tensors=\"pt\"\n",
        "    )\n",
        "\n",
        "    # Ensure the labels are aligned with the inputs (the model expects labels to have the same length as inputs)\n",
        "    # Both the inputs and labels should have the same length after padding\n",
        "    input_encodings['labels'] = target_encodings['input_ids']\n",
        "\n",
        "    # **Change:** Remove the squeeze operation to maintain batch dimensions\n",
        "    # return {k: v.squeeze() for k, v in input_encodings.items()}\n",
        "    return input_encodings # Return the dictionary as is"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TO7mb3ue64US"
      },
      "outputs": [],
      "source": [
        "def preprocess_data(examples):\n",
        "    inputs = examples['input']\n",
        "    responses = examples['response']\n",
        "\n",
        "    # Tokenize the inputs and responses with padding and truncation\n",
        "    input_encodings = tokenizer(inputs, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "    target_encodings = tokenizer(responses, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "\n",
        "    # Ensure the labels are aligned with the inputs (the model expects labels to have the same length as inputs)\n",
        "    input_encodings['labels'] = target_encodings['input_ids']\n",
        "\n",
        "    # Ensure that input and labels are correctly aligned and don't have extra batch dimensions\n",
        "    return {k: v.squeeze() for k, v in input_encodings.items()}\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "E2U8hRol67T2"
      },
      "outputs": [],
      "source": [
        "def preprocess_data(examples):\n",
        "    inputs = examples['input']\n",
        "    responses = examples['response']\n",
        "\n",
        "    # Tokenize the inputs and responses with padding and truncation\n",
        "    input_encodings = tokenizer(inputs, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "    target_encodings = tokenizer(responses, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "\n",
        "    # Ensure the labels are aligned with the inputs (the model expects labels to have the same length as inputs)\n",
        "    input_encodings['labels'] = target_encodings['input_ids']\n",
        "\n",
        "    # Debug: Print the shapes of input_ids and labels\n",
        "    print(f\"Input shape: {input_encodings['input_ids'].shape}\")\n",
        "    print(f\"Labels shape: {input_encodings['labels'].shape}\")\n",
        "\n",
        "    # Ensure that input and labels are correctly aligned and don't have extra batch dimensions\n",
        "    return {k: v.squeeze() for k, v in input_encodings.items()}\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5hu6QCbB6_hH"
      },
      "outputs": [],
      "source": [
        "# Adjust the TrainingArguments to use eval_strategy instead of evaluation_strategy\n",
        "training_args = TrainingArguments(\n",
        "    output_dir=\"./chatbot_model\",  # Directory to save the model\n",
        "    eval_strategy=\"no\",  # Corrected eval_strategy\n",
        "    learning_rate=5e-5,  # Learning rate for training\n",
        "    per_device_train_batch_size=4,  # Batch size for training\n",
        "    num_train_epochs=3,  # Number of training epochs\n",
        "    save_steps=1000,  # Save the model every 1000 steps\n",
        "    logging_dir='./logs',  # Directory for logging\n",
        ")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sslz5Qik7OKe"
      },
      "outputs": [],
      "source": [
        "def preprocess_data(examples):\n",
        "    inputs = examples['input']\n",
        "    responses = examples['response']\n",
        "\n",
        "    # Tokenize inputs with padding and truncation, both for inputs and responses\n",
        "    input_encodings = tokenizer(inputs, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "    target_encodings = tokenizer(responses, truncation=True, padding='max_length', max_length=50, return_tensors=\"pt\")\n",
        "\n",
        "    # Align input_encodings with target_encodings to make sure labels are in the right shape\n",
        "    input_encodings['labels'] = target_encodings['input_ids']\n",
        "\n",
        "    # Return the encoded input and target\n",
        "    return input_encodings\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "P7l1vpX97Q1D"
      },
      "outputs": [],
      "source": [
        "from transformers import DataCollatorForSeq2Seq\n",
        "\n",
        "data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 87,
          "referenced_widgets": [
            "42b3e8391c4f4621864eea7e949420bc",
            "8bd6d5e9bf11447caa007bd86003c5b1",
            "3d269a01672b493aac5ddde9ad5ce697",
            "b300f837f2da425bb9ecd0477676512a",
            "aa7236e6297e4e3cbf7ffcd973b2aa8f",
            "8fbdfe7f05394ddc831e856a1593942c",
            "a7b6c066853d4756824d7f8dabf37409",
            "e15c6783315549b997fc265d422421bd",
            "995c249c4def48c49c24555a8b8b045a",
            "38e6d6afc4a54816929c6b7c1c377ceb",
            "650e7741c95a4111aae655e8e2bc321a"
          ]
        },
        "id": "HVQL0CtK7Tp9",
        "outputId": "7ea496c0-dc1f-4b58-b55f-83930dff7ca3"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "42b3e8391c4f4621864eea7e949420bc",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/1000 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "{'input': 'I’ve left everything for the last minute!', 'response': 'It’s okay to feel like that sometimes. Take a break, talk to someone you trust, and remember that challenges are part of growth. You’re not alone.', 'input_ids': [40, 447, 247, 303, 1364, 2279, 329, 262, 938, 5664, 0, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'labels': [1026, 447, 247, 82, 8788, 284, 1254, 588, 326, 3360, 13, 7214, 257, 2270, 11, 1561, 284, 2130, 345, 3774, 11, 290, 3505, 326, 6459, 389, 636, 286, 3349, 13, 921, 447, 247, 260, 407, 3436, 13, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256]}\n"
          ]
        }
      ],
      "source": [
        "# Load and preprocess the dataset\n",
        "train_dataset = dataset[\"train\"].map(preprocess_data, batched=True)\n",
        "\n",
        "# Review the first example\n",
        "print(train_dataset[0])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_0EnQwiD7Ye2"
      },
      "outputs": [],
      "source": [
        "training_args = TrainingArguments(\n",
        "    output_dir=\"./chatbot_model\",  # Directory to save the model\n",
        "    evaluation_strategy=\"no\",  # Corrected eval_strategy\n",
        "    learning_rate=5e-5,  # Learning rate for training\n",
        "    per_device_train_batch_size=4,  # Batch size for training\n",
        "    num_train_epochs=3,  # Number of training epochs\n",
        "    save_steps=1000,  # Save the model every 1000 steps\n",
        "    logging_dir='./logs',  # Directory for logging\n",
        ")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 198
        },
        "id": "g0l3Uy8x7dD9",
        "outputId": "888d67de-8f05-4c72-c9cf-61fce63feb19"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "    <div>\n",
              "      \n",
              "      <progress value='750' max='750' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
              "      [750/750 1:04:58, Epoch 3/3]\n",
              "    </div>\n",
              "    <table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              " <tr style=\"text-align: left;\">\n",
              "      <th>Step</th>\n",
              "      <th>Training Loss</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <td>500</td>\n",
              "      <td>2.618500</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table><p>"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/plain": [
              "('./chatbot_model/tokenizer_config.json',\n",
              " './chatbot_model/special_tokens_map.json',\n",
              " './chatbot_model/vocab.json',\n",
              " './chatbot_model/merges.txt',\n",
              " './chatbot_model/added_tokens.json')"
            ]
          },
          "execution_count": 28,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Trainer\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    train_dataset=train_dataset,\n",
        "    data_collator=data_collator,\n",
        ")\n",
        "\n",
        "# Start fine-tuning\n",
        "trainer.train()\n",
        "\n",
        "# Save the fine-tuned model\n",
        "model.save_pretrained(\"./chatbot_model\")\n",
        "tokenizer.save_pretrained(\"./chatbot_model\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Zq1CoDgkM5rI"
      },
      "outputs": [],
      "source": [
        "# Ensure this is the correct path to your model directory\n",
        "model_path = \"/content/drive/MyDrive/your_model_folder/chatbot_intent_model\"  # Update with the actual path\n",
        "tokenizer_path = \"/content/drive/MyDrive/your_model_folder/chatbot_intent_model\"\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JzuHTVs7M-K4",
        "outputId": "5c144029-1192-4578-da21-be6989309daa"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model directory not found at ./chatbot_intent_model. Please check the path.\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "\n",
        "# Check if the directory exists and contains the model files\n",
        "model_path = \"./chatbot_intent_model\"\n",
        "if os.path.exists(model_path) and os.path.isdir(model_path):\n",
        "    print(f\"Model directory found at {model_path}\")\n",
        "else:\n",
        "    print(f\"Model directory not found at {model_path}. Please check the path.\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rRBjijskNLZF",
        "outputId": "05209a97-8585-46f6-e091-4f056fe5d7fd"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Current Working Directory: /content\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "\n",
        "# Check the current working directory\n",
        "print(\"Current Working Directory:\", os.getcwd())\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "n6-P4E69NOl8",
        "outputId": "03516406-57b8-4d43-ae01-a3e89f298bd3"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n",
            "Model directory not found at /content/drive/MyDrive/chatbot_intent_model. Please check the path.\n"
          ]
        }
      ],
      "source": [
        "# Check if the model folder is in Google Drive (if you're using Google Colab)\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Check the model folder in your Google Drive\n",
        "model_path = '/content/drive/MyDrive/chatbot_intent_model'\n",
        "if os.path.exists(model_path) and os.path.isdir(model_path):\n",
        "    print(f\"Model directory found at {model_path}\")\n",
        "else:\n",
        "    print(f\"Model directory not found at {model_path}. Please check the path.\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "R1tCKzQvNeWP",
        "outputId": "8e0f2748-56cf-4480-aa83-edae18ccc2d8"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "('./chatbot_intent_model/tokenizer_config.json',\n",
              " './chatbot_intent_model/special_tokens_map.json',\n",
              " './chatbot_intent_model/vocab.json',\n",
              " './chatbot_intent_model/merges.txt',\n",
              " './chatbot_intent_model/added_tokens.json')"
            ]
          },
          "execution_count": 35,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Re-save the model and tokenizer\n",
        "model.save_pretrained('./chatbot_intent_model')\n",
        "tokenizer.save_pretrained('./chatbot_intent_model')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kbdKYvJJN__j",
        "outputId": "158c6abd-ea76-41b0-8419-f2dfdb4056a0"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "['model.safetensors', 'vocab.json', 'merges.txt', 'generation_config.json', 'special_tokens_map.json', 'config.json', 'tokenizer_config.json']\n"
          ]
        }
      ],
      "source": [
        "# Re-save the model and tokenizer with the correct files\n",
        "model.save_pretrained('./chatbot_intent_model')\n",
        "tokenizer.save_pretrained('./chatbot_intent_model')\n",
        "\n",
        "# Check if the files are now saved correctly\n",
        "import os\n",
        "print(os.listdir('./chatbot_intent_model'))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ksUuzMmHOEn0",
        "outputId": "389308c4-ca87-42eb-f3b6-97d0fd0c0d21"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Some weights of GPT2ForSequenceClassification were not initialized from the model checkpoint at ./chatbot_intent_model and are newly initialized: ['score.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "GPT2ForSequenceClassification(\n",
            "  (transformer): GPT2Model(\n",
            "    (wte): Embedding(50257, 768)\n",
            "    (wpe): Embedding(1024, 768)\n",
            "    (drop): Dropout(p=0.1, inplace=False)\n",
            "    (h): ModuleList(\n",
            "      (0-11): 12 x GPT2Block(\n",
            "        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "        (attn): GPT2SdpaAttention(\n",
            "          (c_attn): Conv1D(nf=2304, nx=768)\n",
            "          (c_proj): Conv1D(nf=768, nx=768)\n",
            "          (attn_dropout): Dropout(p=0.1, inplace=False)\n",
            "          (resid_dropout): Dropout(p=0.1, inplace=False)\n",
            "        )\n",
            "        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "        (mlp): GPT2MLP(\n",
            "          (c_fc): Conv1D(nf=3072, nx=768)\n",
            "          (c_proj): Conv1D(nf=768, nx=3072)\n",
            "          (act): NewGELUActivation()\n",
            "          (dropout): Dropout(p=0.1, inplace=False)\n",
            "        )\n",
            "      )\n",
            "    )\n",
            "    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "  )\n",
            "  (score): Linear(in_features=768, out_features=2, bias=False)\n",
            ")\n",
            "GPT2TokenizerFast(name_or_path='./chatbot_intent_model', vocab_size=50257, model_max_length=1024, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={\n",
            "\t50256: AddedToken(\"<|endoftext|>\", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),\n",
            "}\n"
          ]
        }
      ],
      "source": [
        "from transformers import AutoModelForSequenceClassification, AutoTokenizer\n",
        "\n",
        "# Load the model and tokenizer\n",
        "model = AutoModelForSequenceClassification.from_pretrained('./chatbot_intent_model')\n",
        "tokenizer = AutoTokenizer.from_pretrained('./chatbot_intent_model')\n",
        "\n",
        "# Check if the model and tokenizer are loaded successfully\n",
        "print(model)\n",
        "print(tokenizer)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "07MFxU2jOQQH",
        "outputId": "e5998c5d-f0ad-4616-dcda-14a87213ac61"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Some weights of GPT2ForSequenceClassification were not initialized from the model checkpoint at ./chatbot_intent_model and are newly initialized: ['score.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "GPT2ForSequenceClassification(\n",
            "  (transformer): GPT2Model(\n",
            "    (wte): Embedding(50257, 768)\n",
            "    (wpe): Embedding(1024, 768)\n",
            "    (drop): Dropout(p=0.1, inplace=False)\n",
            "    (h): ModuleList(\n",
            "      (0-11): 12 x GPT2Block(\n",
            "        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "        (attn): GPT2SdpaAttention(\n",
            "          (c_attn): Conv1D(nf=2304, nx=768)\n",
            "          (c_proj): Conv1D(nf=768, nx=768)\n",
            "          (attn_dropout): Dropout(p=0.1, inplace=False)\n",
            "          (resid_dropout): Dropout(p=0.1, inplace=False)\n",
            "        )\n",
            "        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "        (mlp): GPT2MLP(\n",
            "          (c_fc): Conv1D(nf=3072, nx=768)\n",
            "          (c_proj): Conv1D(nf=768, nx=3072)\n",
            "          (act): NewGELUActivation()\n",
            "          (dropout): Dropout(p=0.1, inplace=False)\n",
            "        )\n",
            "      )\n",
            "    )\n",
            "    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)\n",
            "  )\n",
            "  (score): Linear(in_features=768, out_features=2, bias=False)\n",
            ")\n",
            "GPT2TokenizerFast(name_or_path='./chatbot_intent_model', vocab_size=50257, model_max_length=1024, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={\n",
            "\t50256: AddedToken(\"<|endoftext|>\", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),\n",
            "}\n"
          ]
        }
      ],
      "source": [
        "from transformers import AutoModelForSequenceClassification, AutoTokenizer\n",
        "\n",
        "# Load the model and tokenizer\n",
        "model = AutoModelForSequenceClassification.from_pretrained('./chatbot_intent_model')\n",
        "tokenizer = AutoTokenizer.from_pretrained('./chatbot_intent_model')\n",
        "\n",
        "# Check if the model and tokenizer are loaded successfully\n",
        "print(model)\n",
        "print(tokenizer)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "utCUbE-MOVdy"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "\n",
        "# Reverse mapping for intent labels (update this with your intent labels)\n",
        "id_to_intent = {0: \"anxiety_management\", 1: \"motivation\"}  # Example intent labels\n",
        "\n",
        "def get_intent(user_input):\n",
        "    # Tokenize the input text (just like training)\n",
        "    tokens = tokenizer(user_input, return_tensors=\"pt\", truncation=True, padding=\"max_length\", max_length=128)\n",
        "\n",
        "    # Make the prediction\n",
        "    with torch.no_grad():\n",
        "        outputs = model(**tokens)\n",
        "\n",
        "    # Get the predicted label (intent)\n",
        "    predicted_label = torch.argmax(outputs.logits, dim=1).item()\n",
        "    return id_to_intent.get(predicted_label, \"Unknown\")  # Map numeric label to the actual intent\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qK3afEjxOY1u",
        "outputId": "526a675c-8e1e-4042-da85-a08a0088bc91"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Predicted Intent: anxiety_management\n"
          ]
        }
      ],
      "source": [
        "# Test the function with some sample inputs\n",
        "user_input = \"I'm feeling anxious about my upcoming exam.\"\n",
        "predicted_intent = get_intent(user_input)\n",
        "print(f\"Predicted Intent: {predicted_intent}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rgoO5nLLQe7P",
        "outputId": "e4a2e0db-dc00-41aa-b7d1-b61de73773cd"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Expanded dataset saved successfully.\n"
          ]
        }
      ],
      "source": [
        "import json\n",
        "import random\n",
        "\n",
        "# Original dataset (40 examples as shown above)\n",
        "data = [\n",
        "    {\"intent\": \"greeting\", \"user\": \"Hello!\", \"response\": \"Hi there! How can I help you today?\"},\n",
        "    {\"intent\": \"goodbye\", \"user\": \"Goodbye!\", \"response\": \"Goodbye! Have a wonderful day ahead!\"},\n",
        "    # Add the rest of your intents here...\n",
        "]\n",
        "\n",
        "# Function to generate more examples by modifying user input slightly\n",
        "def expand_dataset(data, num_examples=1000):\n",
        "    expanded_data = []\n",
        "    for entry in data:\n",
        "        for _ in range(num_examples // len(data)):\n",
        "            modified_example = entry.copy()\n",
        "            modified_example['user'] = modify_example(entry['user'])\n",
        "            expanded_data.append(modified_example)\n",
        "    return expanded_data\n",
        "\n",
        "# Function to modify user input slightly (e.g., add more variations)\n",
        "def modify_example(example):\n",
        "    variations = [\n",
        "        example,\n",
        "        example.replace('!', '?'),\n",
        "        example.lower(),\n",
        "        example.upper()\n",
        "    ]\n",
        "    return random.choice(variations)\n",
        "\n",
        "# Generate 1000 examples\n",
        "expanded_data = expand_dataset(data, num_examples=1000)\n",
        "\n",
        "# Save the expanded dataset\n",
        "with open('expanded_chatbot_data.json', 'w') as f:\n",
        "    json.dump(expanded_data, f, indent=2)\n",
        "\n",
        "print(\"Expanded dataset saved successfully.\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-sFQpNLeRxJ7",
        "outputId": "dfdff772-0e11-4a38-ca80-00042e7efcea"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Some weights of GPT2ForSequenceClassification were not initialized from the model checkpoint at ./chatbot_model and are newly initialized: ['score.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ]
        }
      ],
      "source": [
        "from transformers import AutoModelForSequenceClassification, AutoTokenizer\n",
        "import torch\n",
        "\n",
        "# Load the trained model and tokenizer\n",
        "model = AutoModelForSequenceClassification.from_pretrained('./chatbot_model')\n",
        "tokenizer = AutoTokenizer.from_pretrained('./chatbot_model')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-PrV7QfoR0Yt"
      },
      "outputs": [],
      "source": [
        "# Define a mapping from intent labels to human-readable intent names\n",
        "intent_labels = {\n",
        "    0: \"greeting\",\n",
        "    1: \"goodbye\",\n",
        "    2: \"anxiety_management\",\n",
        "    3: \"motivation\",\n",
        "    4: \"stress_relief\",\n",
        "    5: \"time_management\",\n",
        "    # Add more intents as per your training dataset\n",
        "}\n",
        "\n",
        "def get_intent(user_input):\n",
        "    # Tokenize the user input\n",
        "    inputs = tokenizer(user_input, return_tensors=\"pt\", truncation=True, padding=\"max_length\", max_length=128)\n",
        "\n",
        "    # Get model output\n",
        "    with torch.no_grad():\n",
        "        outputs = model(**inputs)\n",
        "\n",
        "    # Predict the intent (get the index of the maximum logit)\n",
        "    predicted_label = torch.argmax(outputs.logits, dim=1).item()\n",
        "\n",
        "    # Map predicted label to intent name\n",
        "    predicted_intent = intent_labels.get(predicted_label, \"Unknown\")\n",
        "    return predicted_intent\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "h-s-61pISfHb"
      },
      "outputs": [],
      "source": [
        "def get_intent(user_input):\n",
        "    # Tokenize the user input\n",
        "    inputs = tokenizer(user_input, return_tensors=\"pt\", truncation=True, padding=\"max_length\", max_length=128)\n",
        "\n",
        "    # Get model output\n",
        "    with torch.no_grad():\n",
        "        outputs = model(**inputs)\n",
        "\n",
        "    # Get the predicted label (index of maximum logit)\n",
        "    predicted_label = torch.argmax(outputs.logits, dim=1).item()\n",
        "\n",
        "    # Map the predicted label to intent\n",
        "    predicted_intent = intent_labels.get(predicted_label, \"Unknown\")\n",
        "    return predicted_intent\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Sbh0GEjySiL7"
      },
      "outputs": [],
      "source": [
        "def get_intent(user_input):\n",
        "    # Tokenize the user input\n",
        "    inputs = tokenizer(user_input, return_tensors=\"pt\", truncation=True, padding=\"max_length\", max_length=128)\n",
        "\n",
        "    # Get model output\n",
        "    with torch.no_grad():\n",
        "        outputs = model(**inputs)\n",
        "\n",
        "    # Print logits to check the raw model predictions\n",
        "    print(\"Logits:\", outputs.logits)\n",
        "\n",
        "    # Get the predicted label (index of maximum logit)\n",
        "    predicted_label = torch.argmax(outputs.logits, dim=1).item()\n",
        "    print(\"Predicted label:\", predicted_label)\n",
        "\n",
        "    # Map the predicted label to intent\n",
        "    predicted_intent = intent_labels.get(predicted_label, \"Unknown\")\n",
        "    return predicted_intent\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "A2MaX5cQXYFu"
      },
      "outputs": [],
      "source": [
        "# Assuming \"bot\" is the column for the responses\n",
        "train_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"bot\"])\n",
        "val_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"bot\"])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qXIQtUqVYf41"
      },
      "outputs": [],
      "source": [
        "# Assuming \"intent\" is the column for the labels\n",
        "train_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"intent\"])\n",
        "val_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"intent\"])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0SiSHWmdYlND",
        "outputId": "2007123d-cce2-4b34-f556-3ae3acf17894"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Train Dataset Columns: ['user', 'bot', 'intent', 'input_ids', 'token_type_ids', 'attention_mask']\n",
            "Validation Dataset Columns: ['user', 'bot', 'intent', 'input_ids', 'token_type_ids', 'attention_mask']\n"
          ]
        }
      ],
      "source": [
        "print(\"Train Dataset Columns:\", train_dataset.column_names)\n",
        "print(\"Validation Dataset Columns:\", val_dataset.column_names)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xesX-DGFYoGi",
        "outputId": "85724864-c401-408a-91ef-fdc17426f718"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Sample from Train Dataset: {'intent': 'anxiety_management', 'input_ids': tensor([  101,  1045,  1521,  1049,  6091,  2055,  1996, 11360,  1010,  2054,\n",
            "         2323,  1045,  2079,  1029,   102,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0])}\n"
          ]
        }
      ],
      "source": [
        "print(\"Sample from Train Dataset:\", train_dataset[0])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2S_cnpSAZB2j",
        "outputId": "8a410ffd-e694-4aee-cebd-f42b548234ae"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Sample from Train Dataset: {'bot': \"It's completely okay to feel nervous. Focus on your breathing or take a short walk to relieve tension.\", 'input_ids': tensor([  101,  1045,  1521,  1049,  6091,  2055,  1996, 11360,  1010,  2054,\n",
            "         2323,  1045,  2079,  1029,   102,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0])}\n"
          ]
        }
      ],
      "source": [
        "print(\"Sample from Train Dataset:\", train_dataset[0])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 99,
          "referenced_widgets": [
            "1cbad58c80fc47d68d5a6cffbf118ea0",
            "aeca09d2255b4806bf23c119f4b79f5f",
            "2261af50888f41ce91e1d3792782254f",
            "cd113e7f134c4d2d9a0df0c092b149c4",
            "dc9986714a164a91a1f82941371f7e28",
            "452d0102516f470082411f25b6ea8145",
            "cd1bec76be0e48d39f0728172f8ae769",
            "b225767c1ea44f98a11bb9b0e3473586",
            "34c99f1b234640cdbfb8eb2bc3187511",
            "429bfb9f034d4e7f8f6dbc115ced478a",
            "400d431a85f34cc3aa3260d83b07f6c4",
            "749c9e768c0f4a029ef3dbdcfb64b8ed",
            "55214a89e4a449f29cf98f67f2ce28bf",
            "3c26df8da94246d98f3f4e665aa578a5",
            "b40e82b7f9054201bdc8be5bd0f4626c",
            "02ffb12291b347dc8eb7e24c045facd3",
            "6daf03bb5b8d4cad9de207c89b8e159e",
            "d1391f707a2241e9931d28a7a28aa8dc",
            "b9a9a1975883475f912d39f953f99ef0",
            "b37ae495a68143618967005297e86357",
            "f76d4a1bbdb94d44bfa09375d0bbd099",
            "b2c653b79f1f486eb461424e6ec9ddb6"
          ]
        },
        "id": "YfJopK3GTRIL",
        "outputId": "974710dc-5ad3-41f5-ee02-282335f0b01c"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "1cbad58c80fc47d68d5a6cffbf118ea0",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/800 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "749c9e768c0f4a029ef3dbdcfb64b8ed",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/200 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "import json\n",
        "from sklearn.model_selection import train_test_split\n",
        "from datasets import Dataset\n",
        "from transformers import AutoTokenizer\n",
        "\n",
        "# Mount Google Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "import json\n",
        "from sklearn.model_selection import train_test_split\n",
        "from datasets import Dataset\n",
        "from transformers import AutoTokenizer\n",
        "\n",
        "\n",
        "# Load the Study Wellness chatbot dataset\n",
        "# Assuming the file is in your 'My Drive' folder\n",
        "file_path = '/content/drive/My Drive/study_wellness_chatbot_dataset_1000.json'\n",
        "with open(file_path, 'r') as f:\n",
        "    data = json.load(f)\n",
        "\n",
        "# Split the data into training and validation sets (80% training, 20% validation)\n",
        "train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)\n",
        "\n",
        "# Convert to Hugging Face Dataset format\n",
        "train_dataset = Dataset.from_list(train_data)\n",
        "val_dataset = Dataset.from_list(val_data)\n",
        "\n",
        "# Load the tokenizer (BERT for this example)\n",
        "tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n",
        "\n",
        "# Tokenization function\n",
        "def tokenize_function(example):\n",
        "    return tokenizer(example[\"user\"], truncation=True, padding=\"max_length\", max_length=128)\n",
        "\n",
        "train_dataset = train_dataset.map(tokenize_function, batched=True)\n",
        "val_dataset = val_dataset.map(tokenize_function, batched=True)\n",
        "\n",
        "# Prepare the datasets for PyTorch (set the format)\n",
        "# Change 'response' to 'bot' to match the column name in your dataset\n",
        "train_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"bot\"])\n",
        "val_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"bot\"])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 776,
          "referenced_widgets": [
            "83e10f4831c545b79414094a4bbe6fcd",
            "d087616599954d7ea653f17a358edba8",
            "f895362ff17840e6a85fcf220e5aab6d",
            "274db7a30168456a821c0330c7a16d67",
            "e165670cd963463d90c56a3b056e12dc",
            "fc3d94242522474a94dbb2d90a0ed22f",
            "d3d80fcf0cbc491ca3d8dd7f9ef51ecf",
            "ed5b2edeb79549789cd4aac2a95a6a03",
            "62d3a148c4c04892bc112e03445d9cf3",
            "d241cfb7ddf24254be90d25eb51d35d7",
            "44b411a900364a14bfffb50f8cdefaa8",
            "f090ab4c142845faad2fb0ae44304160",
            "aebd0a2a384b42fd96b0b532e305a7db",
            "ab8a8de99e7f4cdc8b5bafcc43c91626",
            "702205a3b42a4e31a858994cfe138b6b",
            "b64792673e4b45908935df2618f29763",
            "58e00ef39a89465fb3fedabf3ff32f58",
            "11908c02fc594707ba856a0b19f54a39",
            "97a7d6c4e07e40c1b060f87a31d07438",
            "3aae5fea0fbe4196996b9942db2f71db",
            "599321f00c9f481888f6e64945dc9d41",
            "7f77db9721154629af5a4a10e137bd5b"
          ]
        },
        "id": "iNClIgzKZfrE",
        "outputId": "e67d9d35-9a24-4131-bc93-b95153f0e37d"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "83e10f4831c545b79414094a4bbe6fcd",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/800 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "f090ab4c142845faad2fb0ae44304160",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/200 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Train Dataset Columns: ['user', 'bot', 'intent', 'input_ids', 'token_type_ids', 'attention_mask']\n",
            "Validation Dataset Columns: ['user', 'bot', 'intent', 'input_ids', 'token_type_ids', 'attention_mask']\n",
            "Sample from Train Dataset: {'bot': \"It's completely okay to feel nervous. Focus on your breathing or take a short walk to relieve tension.\", 'input_ids': tensor([  101,  1045,  1521,  1049,  6091,  2055,  1996, 11360,  1010,  2054,\n",
            "         2323,  1045,  2079,  1029,   102,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,\n",
            "            0,     0,     0,     0,     0,     0,     0,     0]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0])}\n",
            "Sample from Validation Dataset: {'bot': 'It’s okay. Take a deep breath, prioritize your tasks, and focus on the most important ones first.', 'input_ids': tensor([ 101, 1045, 2514, 2066, 1045, 1521, 1049, 2770, 2041, 1997, 2051, 2000,\n",
            "        2817,  102,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
            "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
            "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
            "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
            "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
            "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
            "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
            "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
            "           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,\n",
            "           0,    0,    0,    0,    0,    0,    0,    0]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
            "        0, 0, 0, 0, 0, 0, 0, 0])}\n"
          ]
        }
      ],
      "source": [
        "import json\n",
        "from sklearn.model_selection import train_test_split\n",
        "from datasets import Dataset\n",
        "from transformers import AutoTokenizer\n",
        "\n",
        "# Mount Google Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Load the Study Wellness chatbot dataset\n",
        "file_path = '/content/drive/My Drive/study_wellness_chatbot_dataset_1000.json'\n",
        "with open(file_path, 'r') as f:\n",
        "    data = json.load(f)\n",
        "\n",
        "# Split the data into training and validation sets (80% training, 20% validation)\n",
        "train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)\n",
        "\n",
        "# Convert to Hugging Face Dataset format\n",
        "train_dataset = Dataset.from_list(train_data)\n",
        "val_dataset = Dataset.from_list(val_data)\n",
        "\n",
        "# Load the tokenizer (BERT for this example)\n",
        "tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n",
        "\n",
        "# Tokenization function\n",
        "def tokenize_function(example):\n",
        "    return tokenizer(example[\"user\"], truncation=True, padding=\"max_length\", max_length=128)\n",
        "\n",
        "# Apply tokenization\n",
        "train_dataset = train_dataset.map(tokenize_function, batched=True)\n",
        "val_dataset = val_dataset.map(tokenize_function, batched=True)\n",
        "\n",
        "# Check available columns\n",
        "print(\"Train Dataset Columns:\", train_dataset.column_names)\n",
        "print(\"Validation Dataset Columns:\", val_dataset.column_names)\n",
        "\n",
        "# Prepare the datasets for PyTorch (set the format)\n",
        "# Assuming \"bot\" column contains the labels/responses\n",
        "train_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"bot\"])\n",
        "val_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"bot\"])\n",
        "\n",
        "# Inspect the datasets to ensure proper format\n",
        "print(\"Sample from Train Dataset:\", train_dataset[0])\n",
        "print(\"Sample from Validation Dataset:\", val_dataset[0])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 17,
          "referenced_widgets": [
            "e26e39930f1942aab183ce7578330784",
            "e270679ad8fe4407a62e50e671e95fc0"
          ]
        },
        "id": "MFPghSaUaUq0",
        "outputId": "1b4406d5-61e6-4ab5-aa51-2148cce80dd3"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "e26e39930f1942aab183ce7578330784",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "VBox(children=(HTML(value='<center> <img\\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "!pip install huggingface_hub\n",
        "from huggingface_hub import notebook_login\n",
        "notebook_login()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mA2-zdUMbkuw",
        "outputId": "b875225a-3270-4619-e903-535d0fff8954"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Embedding(50257, 768)"
            ]
          },
          "execution_count": 75,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "!pip install transformers datasets\n",
        "\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "# Load GPT-2 model and tokenizer using AutoClasses\n",
        "model = AutoModelForCausalLM.from_pretrained(\"gpt2\")\n",
        "tokenizer = AutoTokenizer.from_pretrained(\"gpt2\")\n",
        "\n",
        "# Add a special token if needed (e.g., for separating user and bot messages)\n",
        "special_token = \"<|endoftext|>\"\n",
        "tokenizer.add_special_tokens({\"pad_token\": special_token})\n",
        "model.resize_token_embeddings(len(tokenizer))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "B4-L0caSbwZM"
      },
      "outputs": [],
      "source": [
        "# Combine user input and bot responses into one text\n",
        "formatted_data = []\n",
        "for example in data:\n",
        "    user_input = example[\"user\"]\n",
        "    bot_response = example[\"bot\"]\n",
        "    formatted_data.append(f\"{user_input}{special_token}{bot_response}{special_token}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 49,
          "referenced_widgets": [
            "34abdb1f8bfa48939c621eea27e625ee",
            "21ab70a692a74ab3b61c1ea2c74ed3c5",
            "3192dce3a9ae4f60b1b289d8170a0f00",
            "16884993e4734bcabd7a44cd7e6a4068",
            "c5cefc4e110e4c09937bdf7c379317eb",
            "56271191644245ef8647491effd47da2",
            "f0094929c546446fa60146d868fafe41",
            "e9754edfedb442b0979b9d734796c23c",
            "a44971847ef64d07bd611f76e609a1c3",
            "84f49143155141189418c317de1e37a0",
            "ab56c0c791864cedb30bc50c059501fd"
          ]
        },
        "id": "uMeesqmEby26",
        "outputId": "67c0b4c0-5aa5-4738-e49d-1172a59583b0"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "34abdb1f8bfa48939c621eea27e625ee",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/1000 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "from datasets import Dataset\n",
        "\n",
        "# Create a Hugging Face Dataset\n",
        "formatted_dataset = Dataset.from_dict({\"text\": formatted_data})\n",
        "\n",
        "# Tokenize the data\n",
        "def tokenize_function(example):\n",
        "    return tokenizer(\n",
        "        example[\"text\"], truncation=True, padding=\"max_length\", max_length=512\n",
        "    )\n",
        "\n",
        "tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)\n",
        "tokenized_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\"])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Zg52ZfkVcB0Z"
      },
      "outputs": [],
      "source": [
        "# Split the dataset into training and validation sets\n",
        "split_datasets = tokenized_dataset.train_test_split(test_size=0.2, seed=42)\n",
        "train_data = split_datasets[\"train\"]\n",
        "val_data = split_datasets[\"test\"]\n",
        "\n",
        "print(\"Training Data Size:\", len(train_data))\n",
        "print(\"Validation Data Size:\", len(val_data))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000,
          "referenced_widgets": [
            "b434bd1b0f7c43298880e9ee99157082",
            "4655a50836744364b3aa9cf8effb1972",
            "4eb78fe04dc046e78cbfccaac7c3f015",
            "87cb6cc946564673a9848bd5ee69033c",
            "fde47086348640fd86d43757f8de5af5",
            "5b7e0ad9c40d43abb52bb592e997b197",
            "86e38b472d7c4fababbae0b3d5e267cb",
            "68773fad9d6e40569f4f6955e6a1b437",
            "0fa3656a16aa4899ae387e8bb24cfb49",
            "2cad87eee216405bbd8dfec379568a9b",
            "e51cd84ee298440e8f87e163fb0b4bd8",
            "ccbb174dc1ed46b6a9f1784e52120284",
            "47f8aaf3ba6244d8bb90cc6678c32a30",
            "8001fee05cab415daa8ae4a4562dd2f3",
            "35da76c1c0c24b06aa84d1f6b83fd782",
            "4e97eba606eb4f038334c9f08ee08f2b",
            "bdd80cb862884145a3cfc685b3480310",
            "e9f79b49dabc45a8a17d09dc4d3379c4",
            "b7f5ca8f76ef4d16b48bb3f3d6c44604",
            "34f11e67713748b8b26d5be52d457c26",
            "dcaa562147714889944da555eef5a39a",
            "58c1c57bdc234b66854934b9f6245db2",
            "5f5bff86f6354e5b8536f71519696c55",
            "a59a6206590940cbbfe72fc56c2ce20a",
            "a027a7a380274c2b8201fe9529cb00a0",
            "9122b6cd216c4669a20c9d9b6fbd3e01",
            "a12b1d0922514901b35f1abfcb28fe0a",
            "22d9e9afc8c14eb18917279c4768ebf3",
            "4d2ba261f8784e27b08402004e6b4d17",
            "6a3e25ebb8e9428999018d5346f0312c",
            "c20831b89da24fef985c65d16d8314c9",
            "4e64bca526e1457c91f3d78d31a28c0e",
            "477a5ad7c3894cc690d263e09d4485dd",
            "d2279efe79084bafba625c3733a1d825",
            "5a757873adad4feba890b0d3b01c76e1",
            "ba5940168cbf4badb8e1aea2adb7965e",
            "9966803aa4f8434284df70dc65d1df93",
            "715a7ed221b84970a76ec01e9b8c2251",
            "d414bbc106ca4c10b8ebd5e3b96421da",
            "7cf2adc7aa5744e7b92dacbd05707797",
            "4f970a7bb13f461084c9a19335300961",
            "5f8fb54f9c6f49aa83c8ebae33844059",
            "9cdcf372239b405d8ade6ba8f836ff9a",
            "81d4531bc53646c0837de049e7e5fbf5",
            "f7662da059d54f27ab620b474dff0650",
            "8d6088d9f6634812a6bb9f7147125df0",
            "dc44a1f2aa83478f8df181799ab60739",
            "dea2ffabb6304887964eb1896fab372f",
            "ab42d370004d4b38a17842445e21a8f7",
            "7ed618af34d14a81870b599adbf9e81e",
            "c54ea00f47bc449fb68c1f445803b003",
            "573ad370dbe04dcab40b4b7aad6cf305",
            "91b0a6dcfa9f4cb688bef39992af6a2d",
            "cdf49b7e09dc41ddbcb3e7166d0335cf",
            "5173fd3a511f432ca6e70fb2678cc96a",
            "9089dad9898344d983f57adc8af08915",
            "43716ace12be4aa9a8839c50cb44aece",
            "e03368f9c21a4c67a0d750c6be80d944",
            "59899125b3854783a4b2b12fab8ee696",
            "8a8d4b5cd1bd4992bd3c445ba5125c0b",
            "0375039625694d959478b310c332682d",
            "ef6460b1ba2f4c85b18e19dc1c442696",
            "a6cb00ae08934b3fb6c9e2c7281abe73",
            "027ea360b5dd4ea7927718572e82f6ec",
            "f454d3cd4f654ae1892918083c53848c",
            "59903b3255934075962d13952d12ac34",
            "8b40e529989747bf801eb200ce4446f4",
            "58af7b6f315744f1b8f3564df9665e76",
            "381aef476bb941fb9cc79c4812622475",
            "fb961aaa596b44ea90f28b13f2008aaf",
            "5a02aad4c21648a5838382f712933823",
            "845df555f1464f5ba031590837dc555e",
            "6a58411fc04d44cfac845eb131aa4d86",
            "a7bda3966a4d4b7f9b3ffe0d7f04a9c5",
            "70593da4e66d40a6abda62e1f573cddf",
            "784fdfcc81144072a0d252ff96bdae6f",
            "8cf8a605f990424797b0866897a851fa",
            "71e9270f47834a5bb1e039f1b5cd93d4",
            "3ccf553f925b4687a123ca6899ff6ccd",
            "7df4b432da8d405d9df10ff41f47eadd",
            "96ed4c3587844972b2d33dc01efb094a",
            "8ca702a4e3754a99b18978a6c781bb2b",
            "a27d1cb808fb4bcc818123642b45d60c",
            "1f25128e1fe9400aa84b1ea1f86d6d8f",
            "5448634a10934fab99ac696ce7c52d21",
            "5a9d3cef4a404e448216de6532b5aab2",
            "1b3de09bc8f3469d8aca84ab43e27d13",
            "1d9801c7584a4ef382045cb87686a47f"
          ]
        },
        "id": "6ZPPPhmfcRj6",
        "outputId": "e6a5aeb7-5c53-4221-f72d-c86088b5641d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.46.2)\n",
            "Collecting datasets\n",
            "  Downloading datasets-3.1.0-py3-none-any.whl.metadata (20 kB)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.16.1)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.26.2)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.26.4)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.2)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2024.9.11)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.32.3)\n",
            "Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.5)\n",
            "Requirement already satisfied: tokenizers<0.21,>=0.20 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.6)\n",
            "Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (17.0.0)\n",
            "Collecting dill<0.3.9,>=0.3.0 (from datasets)\n",
            "  Downloading dill-0.3.8-py3-none-any.whl.metadata (10 kB)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (2.2.2)\n",
            "Collecting xxhash (from datasets)\n",
            "  Downloading xxhash-3.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)\n",
            "Collecting multiprocess<0.70.17 (from datasets)\n",
            "  Downloading multiprocess-0.70.16-py310-none-any.whl.metadata (7.2 kB)\n",
            "Collecting fsspec<=2024.9.0,>=2023.1.0 (from fsspec[http]<=2024.9.0,>=2023.1.0->datasets)\n",
            "  Downloading fsspec-2024.9.0-py3-none-any.whl.metadata (11 kB)\n",
            "Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.11.2)\n",
            "Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (2.4.3)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (24.2.0)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.5.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.1.0)\n",
            "Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (0.2.0)\n",
            "Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.17.2)\n",
            "Requirement already satisfied: async-timeout<6.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (4.12.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.8.30)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)\n",
            "Downloading datasets-3.1.0-py3-none-any.whl (480 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m480.6/480.6 kB\u001b[0m \u001b[31m15.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading dill-0.3.8-py3-none-any.whl (116 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m116.3/116.3 kB\u001b[0m \u001b[31m9.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading fsspec-2024.9.0-py3-none-any.whl (179 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m179.3/179.3 kB\u001b[0m \u001b[31m3.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading multiprocess-0.70.16-py310-none-any.whl (134 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m134.8/134.8 kB\u001b[0m \u001b[31m4.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading xxhash-3.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m194.1/194.1 kB\u001b[0m \u001b[31m3.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: xxhash, fsspec, dill, multiprocess, datasets\n",
            "  Attempting uninstall: fsspec\n",
            "    Found existing installation: fsspec 2024.10.0\n",
            "    Uninstalling fsspec-2024.10.0:\n",
            "      Successfully uninstalled fsspec-2024.10.0\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "gcsfs 2024.10.0 requires fsspec==2024.10.0, but you have fsspec 2024.9.0 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed datasets-3.1.0 dill-0.3.8 fsspec-2024.9.0 multiprocess-0.70.16 xxhash-3.5.0\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: \n",
            "The secret `HF_TOKEN` does not exist in your Colab secrets.\n",
            "To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.\n",
            "You will be able to reuse this secret in all of your notebooks.\n",
            "Please note that authentication is recommended but still optional to access public models or datasets.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "config.json:   0%|          | 0.00/665 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "b434bd1b0f7c43298880e9ee99157082"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "model.safetensors:   0%|          | 0.00/548M [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "ccbb174dc1ed46b6a9f1784e52120284"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "generation_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "5f5bff86f6354e5b8536f71519696c55"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "tokenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "d2279efe79084bafba625c3733a1d825"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "vocab.json:   0%|          | 0.00/1.04M [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "f7662da059d54f27ab620b474dff0650"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "9089dad9898344d983f57adc8af08915"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "8b40e529989747bf801eb200ce4446f4"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "Map:   0%|          | 0/1000 [00:00<?, ? examples/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "71e9270f47834a5bb1e039f1b5cd93d4"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Data Size: 800\n",
            "Validation Data Size: 200\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/transformers/training_args.py:1568: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead\n",
            "  warnings.warn(\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: \u001b[33mWARNING\u001b[0m The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "        window._wandbApiKey = new Promise((resolve, reject) => {\n",
              "            function loadScript(url) {\n",
              "            return new Promise(function(resolve, reject) {\n",
              "                let newScript = document.createElement(\"script\");\n",
              "                newScript.onerror = reject;\n",
              "                newScript.onload = resolve;\n",
              "                document.body.appendChild(newScript);\n",
              "                newScript.src = url;\n",
              "            });\n",
              "            }\n",
              "            loadScript(\"https://cdn.jsdelivr.net/npm/postmate/build/postmate.min.js\").then(() => {\n",
              "            const iframe = document.createElement('iframe')\n",
              "            iframe.style.cssText = \"width:0;height:0;border:none\"\n",
              "            document.body.appendChild(iframe)\n",
              "            const handshake = new Postmate({\n",
              "                container: iframe,\n",
              "                url: 'https://wandb.ai/authorize'\n",
              "            });\n",
              "            const timeout = setTimeout(() => reject(\"Couldn't auto authenticate\"), 5000)\n",
              "            handshake.then(function(child) {\n",
              "                child.on('authorize', data => {\n",
              "                    clearTimeout(timeout)\n",
              "                    resolve(data)\n",
              "                });\n",
              "            });\n",
              "            })\n",
              "        });\n",
              "    "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mwandb\u001b[0m: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: You can find your API key in your browser here: https://wandb.ai/authorize\n",
            "wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:\n",
            "ERROR:root:Internal Python error in the inspect module.\n",
            "Below is the traceback from this internal error.\n",
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/IPython/core/interactiveshell.py\", line 3553, in run_code\n",
            "    exec(code_obj, self.user_global_ns, self.user_ns)\n",
            "  File \"<ipython-input-1-b4187f113478>\", line 82, in <cell line: 82>\n",
            "    trainer.train()\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/transformers/trainer.py\", line 2123, in train\n",
            "    return inner_training_loop(\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/transformers/trainer.py\", line 2382, in _inner_training_loop\n",
            "    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/transformers/trainer_callback.py\", line 468, in on_train_begin\n",
            "    return self.call_event(\"on_train_begin\", args, state, control)\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/transformers/trainer_callback.py\", line 518, in call_event\n",
            "    result = getattr(callback, event)(\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/transformers/integrations/integration_utils.py\", line 911, in on_train_begin\n",
            "    self.setup(args, state, model, **kwargs)\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/transformers/integrations/integration_utils.py\", line 838, in setup\n",
            "    self._wandb.init(\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_init.py\", line 1270, in init\n",
            "    wandb._sentry.reraise(e)\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/wandb/analytics/sentry.py\", line 161, in reraise\n",
            "    raise exc.with_traceback(sys.exc_info()[2])\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_init.py\", line 1255, in init\n",
            "    wi.setup(kwargs)\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_init.py\", line 304, in setup\n",
            "    wandb_login._login(\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_login.py\", line 347, in _login\n",
            "    wlogin.prompt_api_key()\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_login.py\", line 274, in prompt_api_key\n",
            "    key, status = self._prompt_api_key()\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_login.py\", line 253, in _prompt_api_key\n",
            "    key = apikey.prompt_api_key(\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/wandb/sdk/lib/apikey.py\", line 164, in prompt_api_key\n",
            "    key = input_callback(api_ask).strip()\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/click/termui.py\", line 164, in prompt\n",
            "    value = prompt_func(prompt)\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/click/termui.py\", line 147, in prompt_func\n",
            "    raise Abort() from None\n",
            "click.exceptions.Abort\n",
            "\n",
            "During handling of the above exception, another exception occurred:\n",
            "\n",
            "Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/IPython/core/interactiveshell.py\", line 2099, in showtraceback\n",
            "    stb = value._render_traceback_()\n",
            "AttributeError: 'Abort' object has no attribute '_render_traceback_'\n",
            "\n",
            "During handling of the above exception, another exception occurred:\n",
            "\n",
            "Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/IPython/core/ultratb.py\", line 1101, in get_records\n",
            "    return _fixed_getinnerframes(etb, number_of_lines_of_context, tb_offset)\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/IPython/core/ultratb.py\", line 248, in wrapped\n",
            "    return f(*args, **kwargs)\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/IPython/core/ultratb.py\", line 281, in _fixed_getinnerframes\n",
            "    records = fix_frame_records_filenames(inspect.getinnerframes(etb, context))\n",
            "  File \"/usr/lib/python3.10/inspect.py\", line 1662, in getinnerframes\n",
            "    frameinfo = (tb.tb_frame,) + getframeinfo(tb, context)\n",
            "  File \"/usr/lib/python3.10/inspect.py\", line 1620, in getframeinfo\n",
            "    filename = getsourcefile(frame) or getfile(frame)\n",
            "  File \"/usr/lib/python3.10/inspect.py\", line 829, in getsourcefile\n",
            "    module = getmodule(object, filename)\n",
            "  File \"/usr/lib/python3.10/inspect.py\", line 878, in getmodule\n",
            "    os.path.realpath(f)] = module.__name__\n",
            "  File \"/usr/lib/python3.10/posixpath.py\", line 397, in realpath\n",
            "    return abspath(path)\n",
            "  File \"/usr/lib/python3.10/posixpath.py\", line 386, in abspath\n",
            "    return normpath(path)\n",
            "  File \"/usr/lib/python3.10/posixpath.py\", line 360, in normpath\n",
            "    comps = path.split(sep)\n",
            "KeyboardInterrupt\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "TypeError",
          "evalue": "object of type 'NoneType' has no len()",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAbort\u001b[0m                                     Traceback (most recent call last)",
            "    \u001b[0;31m[... skipping hidden 1 frame]\u001b[0m\n",
            "\u001b[0;32m<ipython-input-1-b4187f113478>\u001b[0m in \u001b[0;36m<cell line: 82>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     81\u001b[0m \u001b[0;31m# Fine-tune GPT-2\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 82\u001b[0;31m \u001b[0mtrainer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     83\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/transformers/trainer.py\u001b[0m in \u001b[0;36mtrain\u001b[0;34m(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)\u001b[0m\n\u001b[1;32m   2122\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2123\u001b[0;31m             return inner_training_loop(\n\u001b[0m\u001b[1;32m   2124\u001b[0m                 \u001b[0margs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/transformers/trainer.py\u001b[0m in \u001b[0;36m_inner_training_loop\u001b[0;34m(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)\u001b[0m\n\u001b[1;32m   2381\u001b[0m         \u001b[0mgrad_norm\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mOptional\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2382\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcontrol\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcallback_handler\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_train_begin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstate\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcontrol\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2383\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/transformers/trainer_callback.py\u001b[0m in \u001b[0;36mon_train_begin\u001b[0;34m(self, args, state, control)\u001b[0m\n\u001b[1;32m    467\u001b[0m         \u001b[0mcontrol\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshould_training_stop\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mFalse\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 468\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcall_event\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"on_train_begin\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mstate\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcontrol\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    469\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/transformers/trainer_callback.py\u001b[0m in \u001b[0;36mcall_event\u001b[0;34m(self, event, args, state, control, **kwargs)\u001b[0m\n\u001b[1;32m    517\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mcallback\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcallbacks\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 518\u001b[0;31m             result = getattr(callback, event)(\n\u001b[0m\u001b[1;32m    519\u001b[0m                 \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/transformers/integrations/integration_utils.py\u001b[0m in \u001b[0;36mon_train_begin\u001b[0;34m(self, args, state, control, model, **kwargs)\u001b[0m\n\u001b[1;32m    910\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_initialized\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 911\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msetup\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mstate\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    912\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/transformers/integrations/integration_utils.py\u001b[0m in \u001b[0;36msetup\u001b[0;34m(self, args, state, model, **kwargs)\u001b[0m\n\u001b[1;32m    837\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_wandb\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrun\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 838\u001b[0;31m                 self._wandb.init(\n\u001b[0m\u001b[1;32m    839\u001b[0m                     \u001b[0mproject\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgetenv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"WANDB_PROJECT\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"huggingface\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_init.py\u001b[0m in \u001b[0;36minit\u001b[0;34m(job_type, dir, config, project, entity, reinit, tags, group, name, notes, magic, config_exclude_keys, config_include_keys, anonymous, mode, allow_val_change, resume, force, tensorboard, sync_tensorboard, monitor_gym, save_code, id, fork_from, resume_from, settings)\u001b[0m\n\u001b[1;32m   1269\u001b[0m         \u001b[0;31m# mess with sentry's ability to send out errors before the program ends.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1270\u001b[0;31m         \u001b[0mwandb\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_sentry\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreraise\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1271\u001b[0m         \u001b[0;32mraise\u001b[0m \u001b[0mAssertionError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# unreachable\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/wandb/analytics/sentry.py\u001b[0m in \u001b[0;36mreraise\u001b[0;34m(self, exc)\u001b[0m\n\u001b[1;32m    160\u001b[0m         \u001b[0;31m# but hopefully it's not too bad\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 161\u001b[0;31m         \u001b[0;32mraise\u001b[0m \u001b[0mexc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwith_traceback\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexc_info\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    162\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_init.py\u001b[0m in \u001b[0;36minit\u001b[0;34m(job_type, dir, config, project, entity, reinit, tags, group, name, notes, magic, config_exclude_keys, config_include_keys, anonymous, mode, allow_val_change, resume, force, tensorboard, sync_tensorboard, monitor_gym, save_code, id, fork_from, resume_from, settings)\u001b[0m\n\u001b[1;32m   1254\u001b[0m         \u001b[0mwi\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_WandbInit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1255\u001b[0;31m         \u001b[0mwi\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msetup\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1256\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mwi\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0minit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_init.py\u001b[0m in \u001b[0;36msetup\u001b[0;34m(self, kwargs)\u001b[0m\n\u001b[1;32m    303\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0msettings\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_offline\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0msettings\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_noop\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 304\u001b[0;31m             wandb_login._login(\n\u001b[0m\u001b[1;32m    305\u001b[0m                 \u001b[0manonymous\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"anonymous\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_login.py\u001b[0m in \u001b[0;36m_login\u001b[0;34m(anonymous, key, relogin, host, force, timeout, _backend, _silent, _disable_warning, _entity)\u001b[0m\n\u001b[1;32m    346\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mkey\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 347\u001b[0;31m         \u001b[0mwlogin\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mprompt_api_key\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    348\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_login.py\u001b[0m in \u001b[0;36mprompt_api_key\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    273\u001b[0m         \u001b[0;34m\"\"\"Updates the global API key by prompting the user.\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 274\u001b[0;31m         \u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mstatus\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_prompt_api_key\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    275\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mstatus\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0mApiKeyStatus\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mNOTTY\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/wandb/sdk/wandb_login.py\u001b[0m in \u001b[0;36m_prompt_api_key\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    252\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 253\u001b[0;31m                 key = apikey.prompt_api_key(\n\u001b[0m\u001b[1;32m    254\u001b[0m                     \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_settings\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/wandb/sdk/lib/apikey.py\u001b[0m in \u001b[0;36mprompt_api_key\u001b[0;34m(settings, api, input_callback, browser_callback, no_offline, no_create, local)\u001b[0m\n\u001b[1;32m    163\u001b[0m             )\n\u001b[0;32m--> 164\u001b[0;31m             \u001b[0mkey\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0minput_callback\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mapi_ask\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstrip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    165\u001b[0m         \u001b[0mwrite_key\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msettings\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkey\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mapi\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mapi\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/click/termui.py\u001b[0m in \u001b[0;36mprompt\u001b[0;34m(text, default, hide_input, confirmation_prompt, type, value_proc, prompt_suffix, show_default, err, show_choices)\u001b[0m\n\u001b[1;32m    163\u001b[0m         \u001b[0;32mwhile\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 164\u001b[0;31m             \u001b[0mvalue\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mprompt_func\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mprompt\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    165\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/click/termui.py\u001b[0m in \u001b[0;36mprompt_func\u001b[0;34m(text)\u001b[0m\n\u001b[1;32m    146\u001b[0m                 \u001b[0mecho\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0merr\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0merr\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 147\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mAbort\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    148\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAbort\u001b[0m: ",
            "\nDuring handling of the above exception, another exception occurred:\n",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/IPython/core/interactiveshell.py\u001b[0m in \u001b[0;36mshowtraceback\u001b[0;34m(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)\u001b[0m\n\u001b[1;32m   2098\u001b[0m                         \u001b[0;31m# in the engines. This should return a list of strings.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2099\u001b[0;31m                         \u001b[0mstb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_render_traceback_\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2100\u001b[0m                     \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAttributeError\u001b[0m: 'Abort' object has no attribute '_render_traceback_'",
            "\nDuring handling of the above exception, another exception occurred:\n",
            "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
            "    \u001b[0;31m[... skipping hidden 1 frame]\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/IPython/core/interactiveshell.py\u001b[0m in \u001b[0;36mshowtraceback\u001b[0;34m(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)\u001b[0m\n\u001b[1;32m   2099\u001b[0m                         \u001b[0mstb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_render_traceback_\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2100\u001b[0m                     \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2101\u001b[0;31m                         stb = self.InteractiveTB.structured_traceback(etype,\n\u001b[0m\u001b[1;32m   2102\u001b[0m                                             value, tb, tb_offset=tb_offset)\n\u001b[1;32m   2103\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, value, tb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[1;32m   1365\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1366\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtb\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1367\u001b[0;31m         return FormattedTB.structured_traceback(\n\u001b[0m\u001b[1;32m   1368\u001b[0m             self, etype, value, tb, tb_offset, number_of_lines_of_context)\n\u001b[1;32m   1369\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, value, tb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[1;32m   1265\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mmode\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mverbose_modes\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1266\u001b[0m             \u001b[0;31m# Verbose modes need a full traceback\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1267\u001b[0;31m             return VerboseTB.structured_traceback(\n\u001b[0m\u001b[1;32m   1268\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtb\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtb_offset\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnumber_of_lines_of_context\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1269\u001b[0m             )\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mstructured_traceback\u001b[0;34m(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[1;32m   1122\u001b[0m         \u001b[0;34m\"\"\"Return a nice text document describing the traceback.\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1123\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1124\u001b[0;31m         formatted_exception = self.format_exception_as_a_whole(etype, evalue, etb, number_of_lines_of_context,\n\u001b[0m\u001b[1;32m   1125\u001b[0m                                                                tb_offset)\n\u001b[1;32m   1126\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mformat_exception_as_a_whole\u001b[0;34m(self, etype, evalue, etb, number_of_lines_of_context, tb_offset)\u001b[0m\n\u001b[1;32m   1080\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1081\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1082\u001b[0;31m         \u001b[0mlast_unique\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecursion_repeat\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfind_recursion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0morig_etype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1083\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1084\u001b[0m         \u001b[0mframes\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mformat_records\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrecords\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlast_unique\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecursion_repeat\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/IPython/core/ultratb.py\u001b[0m in \u001b[0;36mfind_recursion\u001b[0;34m(etype, value, records)\u001b[0m\n\u001b[1;32m    380\u001b[0m     \u001b[0;31m# first frame (from in to out) that looks different.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    381\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mis_recursion_error\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0metype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrecords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 382\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrecords\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    383\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    384\u001b[0m     \u001b[0;31m# Select filename, lineno, func_name to track frames with\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mTypeError\u001b[0m: object of type 'NoneType' has no len()"
          ]
        }
      ],
      "source": [
        "# Install necessary libraries (if not already installed)\n",
        "!pip install transformers datasets\n",
        "\n",
        "# Import necessary libraries\n",
        "import json\n",
        "from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer\n",
        "from datasets import Dataset\n",
        "\n",
        "# Load GPT-2 model and tokenizer\n",
        "model = GPT2LMHeadModel.from_pretrained(\"gpt2\")\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")\n",
        "\n",
        "# Add a special token if needed (e.g., for separating user and bot messages)\n",
        "special_token = \"<|endoftext|>\"\n",
        "tokenizer.add_special_tokens({\"pad_token\": special_token})\n",
        "model.resize_token_embeddings(len(tokenizer))\n",
        "\n",
        "# Mount Google Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Load the Study Wellness chatbot dataset\n",
        "file_path = '/content/drive/My Drive/study_wellness_chatbot_dataset_1000.json'\n",
        "with open(file_path, 'r') as f:\n",
        "    data = json.load(f)\n",
        "\n",
        "# Prepare formatted data\n",
        "# Assuming `data` is your original dataset with \"user\" and \"bot\" keys\n",
        "formatted_data = []\n",
        "for example in data:\n",
        "    user_input = example[\"user\"]\n",
        "    bot_response = example[\"bot\"]\n",
        "    formatted_data.append(f\"{user_input}{special_token}{bot_response}{special_token}\")\n",
        "\n",
        "\n",
        "# Create a Hugging Face Dataset\n",
        "formatted_dataset = Dataset.from_dict({\"text\": formatted_data})\n",
        "\n",
        "# Tokenize the data\n",
        "def tokenize_function(example):\n",
        "    # Tokenize the text and set labels to the input_ids shifted by one position\n",
        "    tokenized = tokenizer(\n",
        "        example[\"text\"], truncation=True, padding=\"max_length\", max_length=512\n",
        "    )\n",
        "    tokenized[\"labels\"] = tokenized[\"input_ids\"].copy()  # Set labels for loss calculation\n",
        "    return tokenized\n",
        "\n",
        "tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)\n",
        "tokenized_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"labels\"]) # Include 'labels'\n",
        "\n",
        "# Split the dataset into training and validation sets\n",
        "split_datasets = tokenized_dataset.train_test_split(test_size=0.2, seed=42)\n",
        "train_data = split_datasets[\"train\"]\n",
        "val_data = split_datasets[\"test\"]\n",
        "\n",
        "print(\"Training Data Size:\", len(train_data))\n",
        "print(\"Validation Data Size:\", len(val_data))\n",
        "\n",
        "# Set up training arguments\n",
        "training_args = TrainingArguments(\n",
        "    output_dir=\"./gpt2_results\",\n",
        "    evaluation_strategy=\"epoch\",\n",
        "    learning_rate=5e-5,\n",
        "    per_device_train_batch_size=8,  # Adjust based on GPU memory\n",
        "    per_device_eval_batch_size=8,\n",
        "    num_train_epochs=3,\n",
        "    weight_decay=0.01,\n",
        "    save_steps=500,\n",
        "    save_total_limit=2,\n",
        "    logging_dir=\"./logs\",\n",
        ")\n",
        "\n",
        "# Initialize the Trainer\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    train_dataset=train_data,\n",
        "    eval_dataset=val_data,\n",
        ")\n",
        "\n",
        "# Fine-tune GPT-2\n",
        "trainer.train()\n",
        "\n",
        "# Save the fine-tuned model\n",
        "model.save_pretrained(\"./fine_tuned_gpt2\")\n",
        "tokenizer.save_pretrained(\"./fine_tuned_gpt2\")\n",
        "\n",
        "# Test the chatbot\n",
        "from transformers import pipeline\n",
        "\n",
        "chatbot = pipeline(\"text-generation\", model=\"./fine_tuned_gpt2\", tokenizer=tokenizer)\n",
        "\n",
        "# Generate a response\n",
        "user_input = \"I feel anxious about my exams.\"\n",
        "response = chatbot(f\"{user_input}{special_token}\", max_length=50, num_return_sequences=1)\n",
        "print(response[0][\"generated_text\"])\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Install necessary libraries (if not already installed)\n",
        "!pip install transformers datasets\n",
        "\n",
        "# Import necessary libraries\n",
        "import json\n",
        "from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer\n",
        "from datasets import Dataset\n",
        "\n",
        "# Load GPT-2 model and tokenizer (Consider using distilgpt2 for faster training)\n",
        "model = GPT2LMHeadModel.from_pretrained(\"gpt2\")\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")\n",
        "\n",
        "# Add a special token if needed (e.g., for separating user and bot messages)\n",
        "special_token = \"<|endoftext|>\"\n",
        "tokenizer.add_special_tokens({\"pad_token\": special_token})\n",
        "model.resize_token_embeddings(len(tokenizer))\n",
        "\n",
        "# Mount Google Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Load the Study Wellness chatbot dataset\n",
        "file_path = '/content/drive/My Drive/study_wellness_chatbot_dataset_1000.json'\n",
        "with open(file_path, 'r') as f:\n",
        "    data = json.load(f)\n",
        "\n",
        "# Prepare formatted data\n",
        "formatted_data = [\n",
        "    f\"{example['user']}{special_token}{example['bot']}{special_token}\" for example in data\n",
        "]\n",
        "\n",
        "# Create a Hugging Face Dataset\n",
        "formatted_dataset = Dataset.from_dict({\"text\": formatted_data})\n",
        "\n",
        "# Tokenize the data\n",
        "def tokenize_function(example):\n",
        "    tokenized = tokenizer(\n",
        "        example[\"text\"], truncation=True, padding=\"max_length\", max_length=256  # Reduced max_length\n",
        "    )\n",
        "    tokenized[\"labels\"] = tokenized[\"input_ids\"].copy()\n",
        "    return tokenized\n",
        "\n",
        "tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)\n",
        "tokenized_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"labels\"])\n",
        "\n",
        "# Split the dataset into training and validation sets\n",
        "split_datasets = tokenized_dataset.train_test_split(test_size=0.2, seed=42)\n",
        "train_data = split_datasets[\"train\"]\n",
        "val_data = split_datasets[\"test\"]\n",
        "\n",
        "print(\"Training Data Size:\", len(train_data))\n",
        "print(\"Validation Data Size:\", len(val_data))\n",
        "\n",
        "# Set up optimized training arguments\n",
        "training_args = TrainingArguments(\n",
        "    output_dir=\"./gpt2_results\",\n",
        "    evaluation_strategy=\"epoch\",\n",
        "    learning_rate=5e-5,\n",
        "    per_device_train_batch_size=8,  # Reduced batch size\n",
        "    per_device_eval_batch_size=8,\n",
        "    num_train_epochs=3,  # Lower epochs for initial testing\n",
        "    weight_decay=0.01,\n",
        "    save_steps=1000,  # Less frequent checkpoints\n",
        "    save_total_limit=2,\n",
        "    logging_dir=\"./logs\",\n",
        "    fp16=True,  # Enable mixed precision training\n",
        "    logging_steps=200,  # Reduced logging frequency\n",
        ")\n",
        "\n",
        "# Initialize the Trainer\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    train_dataset=train_data,\n",
        "    eval_dataset=val_data,\n",
        ")\n",
        "\n",
        "# Fine-tune GPT-2\n",
        "trainer.train()\n",
        "\n",
        "# Save the fine-tuned model\n",
        "model.save_pretrained(\"./fine_tuned_gpt2\")\n",
        "tokenizer.save_pretrained(\"./fine_tuned_gpt2\")\n",
        "\n",
        "# Test the chatbot\n",
        "from transformers import pipeline\n",
        "\n",
        "chatbot = pipeline(\"text-generation\", model=\"./fine_tuned_gpt2\", tokenizer=tokenizer)\n",
        "\n",
        "# Generate a response\n",
        "user_input = \"I feel anxious about my exams.\"\n",
        "response = chatbot(f\"{user_input}{special_token}\", max_length=50, num_return_sequences=1)\n",
        "print(response[0][\"generated_text\"])\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000,
          "referenced_widgets": [
            "bf916e995bd54c83b523f06908cf0642",
            "ef8ad31cfe6b4cd2abd023413ef8e97f",
            "69f081fb37ff4c6f853545b49b179106",
            "1df417f3c1d94a2e8a9514f8b501e90a",
            "27b6db172e3a45b1a6a2dc7e59a3bdc6",
            "f2378fc6533d4d37900b75c25b932452",
            "2d2ea78ac132401dbd55c7dbe08fa266",
            "cfa05e787eb04406b76936f1104be832",
            "008f7221bef74bdcadc5b06ecd0915f3",
            "46c99ea38fa6477699566077554d6347",
            "bafd0dc85c6b429382bba264ca85e4e1"
          ]
        },
        "id": "k_WwWQdTF9KX",
        "outputId": "d23dc761-374a-44a7-9965-f85cd463dc42"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.46.2)\n",
            "Requirement already satisfied: datasets in /usr/local/lib/python3.10/dist-packages (3.1.0)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.16.1)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.26.2)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.26.4)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.2)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2024.9.11)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.32.3)\n",
            "Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.5)\n",
            "Requirement already satisfied: tokenizers<0.21,>=0.20 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.6)\n",
            "Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (17.0.0)\n",
            "Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.3.8)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (2.2.2)\n",
            "Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets) (3.5.0)\n",
            "Requirement already satisfied: multiprocess<0.70.17 in /usr/local/lib/python3.10/dist-packages (from datasets) (0.70.16)\n",
            "Requirement already satisfied: fsspec<=2024.9.0,>=2023.1.0 in /usr/local/lib/python3.10/dist-packages (from fsspec[http]<=2024.9.0,>=2023.1.0->datasets) (2024.9.0)\n",
            "Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.11.2)\n",
            "Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (2.4.3)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (24.2.0)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.5.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.1.0)\n",
            "Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (0.2.0)\n",
            "Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.17.2)\n",
            "Requirement already satisfied: async-timeout<6.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (4.12.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.8.30)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)\n",
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "Map:   0%|          | 0/1000 [00:00<?, ? examples/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "bf916e995bd54c83b523f06908cf0642"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Data Size: 800\n",
            "Validation Data Size: 200\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/transformers/training_args.py:1568: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "        window._wandbApiKey = new Promise((resolve, reject) => {\n",
              "            function loadScript(url) {\n",
              "            return new Promise(function(resolve, reject) {\n",
              "                let newScript = document.createElement(\"script\");\n",
              "                newScript.onerror = reject;\n",
              "                newScript.onload = resolve;\n",
              "                document.body.appendChild(newScript);\n",
              "                newScript.src = url;\n",
              "            });\n",
              "            }\n",
              "            loadScript(\"https://cdn.jsdelivr.net/npm/postmate/build/postmate.min.js\").then(() => {\n",
              "            const iframe = document.createElement('iframe')\n",
              "            iframe.style.cssText = \"width:0;height:0;border:none\"\n",
              "            document.body.appendChild(iframe)\n",
              "            const handshake = new Postmate({\n",
              "                container: iframe,\n",
              "                url: 'https://wandb.ai/authorize'\n",
              "            });\n",
              "            const timeout = setTimeout(() => reject(\"Couldn't auto authenticate\"), 5000)\n",
              "            handshake.then(function(child) {\n",
              "                child.on('authorize', data => {\n",
              "                    clearTimeout(timeout)\n",
              "                    resolve(data)\n",
              "                });\n",
              "            });\n",
              "            })\n",
              "        });\n",
              "    "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mwandb\u001b[0m: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)\n",
            "\u001b[34m\u001b[1mwandb\u001b[0m: You can find your API key in your browser here: https://wandb.ai/authorize\n",
            "wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            " ··········\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\u001b[34m\u001b[1mwandb\u001b[0m: Appending key for api.wandb.ai to your netrc file: /root/.netrc\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "Tracking run with wandb version 0.18.7"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "Run data is saved locally in <code>/content/wandb/run-20241202_145713-lzipe1hr</code>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "Syncing run <strong><a href='https://wandb.ai/rsharma23121989-yale-university/huggingface/runs/lzipe1hr' target=\"_blank\">./gpt2_results</a></strong> to <a href='https://wandb.ai/rsharma23121989-yale-university/huggingface' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target=\"_blank\">docs</a>)<br/>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              " View project at <a href='https://wandb.ai/rsharma23121989-yale-university/huggingface' target=\"_blank\">https://wandb.ai/rsharma23121989-yale-university/huggingface</a>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              " View run at <a href='https://wandb.ai/rsharma23121989-yale-university/huggingface/runs/lzipe1hr' target=\"_blank\">https://wandb.ai/rsharma23121989-yale-university/huggingface/runs/lzipe1hr</a>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "    <div>\n",
              "      \n",
              "      <progress value='300' max='300' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
              "      [300/300 3:25:01, Epoch 3/3]\n",
              "    </div>\n",
              "    <table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              " <tr style=\"text-align: left;\">\n",
              "      <th>Epoch</th>\n",
              "      <th>Training Loss</th>\n",
              "      <th>Validation Loss</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <td>1</td>\n",
              "      <td>No log</td>\n",
              "      <td>0.016444</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <td>2</td>\n",
              "      <td>0.201600</td>\n",
              "      <td>0.013933</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <td>3</td>\n",
              "      <td>0.201600</td>\n",
              "      <td>0.012822</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table><p>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "I feel anxious about my exams.<|endoftext|>Deep breathing exercises and positive affirmations can help ease anxiety. Want to try one?\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# This code snippet is to create GUI interface for interaction with the\n",
        "!pip install streamlit\n",
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "# Load your fine-tuned GPT-2 model and tokenizer\n",
        "model = AutoModelForCausalLM.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "tokenizer = AutoTokenizer.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "special_token = \"<|endoftext|>\"  # Your special token\n",
        "\n",
        "# Streamlit app title\n",
        "st.title(\"Mental Wellbeing Chatbot\")\n",
        "\n",
        "# User input area\n",
        "user_input = st.text_input(\"You:\")\n",
        "\n",
        "# Generate response button\n",
        "if st.button(\"Generate Response\"):\n",
        "    if user_input:\n",
        "        # Generate response using the model\n",
        "        input_ids = tokenizer.encode(user_input + special_token, return_tensors=\"pt\")\n",
        "        output = model.generate(input_ids, max_length=100, num_return_sequences=1)\n",
        "        response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "\n",
        "        # Display the response\n",
        "        st.write(\"Chatbot:\", response)\n",
        "    else:\n",
        "        st.write(\"Please enter a message.\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EovIGbWyCt_s",
        "outputId": "528b0bb9-d31e-4121-bb2b-10eabee87794"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.40.2-py2.py3-none-any.whl.metadata (8.4 kB)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (11.0.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.25.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (17.0.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.9.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Collecting watchdog<7,>=2.1.5 (from streamlit)\n",
            "  Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl.metadata (44 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m44.3/44.3 kB\u001b[0m \u001b[31m1.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.43)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.8.30)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.18.0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.21.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.16.0)\n",
            "Downloading streamlit-1.40.2-py2.py3-none-any.whl (8.6 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.6/8.6 MB\u001b[0m \u001b[31m29.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m30.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl (79 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m79.1/79.1 kB\u001b[0m \u001b[31m2.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: watchdog, pydeck, streamlit\n",
            "Successfully installed pydeck-0.9.1 streamlit-1.40.2 watchdog-6.0.0\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-12-02 18:24:40.871 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.007 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2024-12-02 18:24:41.010 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.013 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.016 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.018 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.020 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.023 Session state does not function when running a script without `streamlit run`\n",
            "2024-12-02 18:24:41.025 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.028 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.030 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.031 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.033 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.034 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:24:41.035 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit\n",
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "# Load your fine-tuned GPT-2 model and tokenizer\n",
        "@st.cache_resource # Cache the model and tokenizer for faster loading\n",
        "def load_model_and_tokenizer():\n",
        "    \"\"\"Loads and returns the pre-trained model and tokenizer.\"\"\"\n",
        "    model = AutoModelForCausalLM.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    tokenizer = AutoTokenizer.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    return model, tokenizer\n",
        "\n",
        "model, tokenizer = load_model_and_tokenizer()\n",
        "special_token = \"<|endoftext|>\"  # Your special token\n",
        "\n",
        "# Streamlit app title\n",
        "st.title(\"Mental Wellbeing Chatbot\")\n",
        "\n",
        "# User input area\n",
        "user_input = st.text_input(\"You:\")\n",
        "\n",
        "# Generate response button\n",
        "if st.button(\"Generate Response\"):\n",
        "    if user_input:\n",
        "        # Generate response using the model\n",
        "        input_ids = tokenizer.encode(user_input + special_token, return_tensors=\"pt\")\n",
        "        output = model.generate(input_ids, max_length=100, num_return_sequences=1)\n",
        "        response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "\n",
        "        # Display the response\n",
        "        st.write(\"Chatbot:\", response)\n",
        "    else:\n",
        "        st.write(\"Please enter a message.\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lpMURKrenuLb",
        "outputId": "2c717b9b-dc40-43db-c2dc-1a33f5496d4e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.40.2)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (11.0.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.25.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (17.0.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.9.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.43)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.8.30)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.18.0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.21.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.16.0)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-12-02 18:25:43.407 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.414 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.419 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.852 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.861 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.868 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.874 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.881 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.886 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.891 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.898 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.913 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.924 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.926 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.931 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.947 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.956 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:25:43.964 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3NOTQin_n-mD",
        "outputId": "53ff0e87-eaf5-4a18-c035-4fb00fdb9efc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Usage: streamlit run [OPTIONS] TARGET [ARGS]...\n",
            "Try 'streamlit run --help' for help.\n",
            "\n",
            "Error: Invalid value: File does not exist: app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit pyngrok\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mgeAYVRnpQBx",
        "outputId": "2351b882-3d6b-47f7-a8ef-2d67716f1c1c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.40.2)\n",
            "Collecting pyngrok\n",
            "  Downloading pyngrok-7.2.1-py3-none-any.whl.metadata (8.3 kB)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (11.0.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.25.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (17.0.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.9.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.43)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.10/dist-packages (from pyngrok) (6.0.2)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.8.30)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.18.0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.21.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.16.0)\n",
            "Downloading pyngrok-7.2.1-py3-none-any.whl (22 kB)\n",
            "Installing collected packages: pyngrok\n",
            "Successfully installed pyngrok-7.2.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "# Load your fine-tuned model\n",
        "@st.cache_resource\n",
        "def load_model_and_tokenizer():\n",
        "    model = AutoModelForCausalLM.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    tokenizer = AutoTokenizer.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    return model, tokenizer\n",
        "\n",
        "model, tokenizer = load_model_and_tokenizer()\n",
        "\n",
        "st.title(\"Mental Wellbeing Chatbot\")\n",
        "user_input = st.text_input(\"You:\", placeholder=\"Type your message here...\")\n",
        "\n",
        "if st.button(\"Generate Response\"):\n",
        "    if user_input:\n",
        "        input_ids = tokenizer.encode(user_input + \"<|endoftext|>\", return_tensors=\"pt\")\n",
        "        output = model.generate(input_ids, max_length=100, num_return_sequences=1)\n",
        "        response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "        st.write(f\"Chatbot: {response}\")\n",
        "    else:\n",
        "        st.write(\"Please enter a message!\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3Dl8jB06qBQS",
        "outputId": "0304c8f2-34d4-490e-f6c4-93d0d77f3271"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-12-02 18:35:40.739 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:40.742 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:40.745 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.274 Thread 'Thread-19': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.286 Thread 'Thread-19': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.291 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.294 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.305 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.309 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.324 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.330 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.334 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.339 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.343 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.345 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.352 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.354 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.374 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.377 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-02 18:35:41.380 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ls\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "o9joRYeUqE9t",
        "outputId": "0172bb93-9624-429c-919e-45037c267572"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[0m\u001b[01;34mdrive\u001b[0m/  \u001b[01;34mfine_tuned_gpt2\u001b[0m/  \u001b[01;34mgpt2_results\u001b[0m/  \u001b[01;34mlogs\u001b[0m/  \u001b[01;34msample_data\u001b[0m/  \u001b[01;34mwandb\u001b[0m/\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ngrok config add-authtoken YOUR_AUTHTOKEN\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "wO5w9lOxqJ49",
        "outputId": "e4ca381e-5ce5-412b-b45e-8b2c731550aa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Authtoken saved to configuration file: /root/.config/ngrok/ngrok.yml\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vhr60vDuq0Rf",
        "outputId": "cd79f354-66cb-4d73-ff66-cfd0fe9b2603"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "with open('/content/drive/My Drive/app.py', 'w') as f:\n",
        "    f.write(\"\"\"\n",
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "@st.cache_resource\n",
        "def load_model_and_tokenizer():\n",
        "    model = AutoModelForCausalLM.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    tokenizer = AutoTokenizer.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    return model, tokenizer\n",
        "\n",
        "model, tokenizer = load_model_and_tokenizer()\n",
        "\n",
        "st.title(\"Mental Wellbeing Chatbot\")\n",
        "user_input = st.text_input(\"You:\", placeholder=\"Type your message here...\")\n",
        "\n",
        "if st.button(\"Generate Response\"):\n",
        "    if user_input:\n",
        "        input_ids = tokenizer.encode(user_input + \"<|endoftext|>\", return_tensors=\"pt\")\n",
        "        output = model.generate(input_ids, max_length=100, num_return_sequences=1)\n",
        "        response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "        st.write(f\"Chatbot: {response}\")\n",
        "    else:\n",
        "        st.write(\"Please enter a message!\")\n",
        "\"\"\")\n"
      ],
      "metadata": {
        "id": "ZosX7O8Pq3Qc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 464
        },
        "id": "U0PG3RoUsryP",
        "outputId": "a2b22b39-e36e-4982-a6a9-251ec9c3d9e6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "ERROR:pyngrok.process.ngrok:t=2024-12-02T18:47:30+0000 lvl=eror msg=\"failed to reconnect session\" obj=tunnels.session err=\"authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n\"\n",
            "ERROR:pyngrok.process.ngrok:t=2024-12-02T18:47:30+0000 lvl=eror msg=\"session closing\" obj=tunnels.session err=\"authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n\"\n",
            "ERROR:pyngrok.process.ngrok:t=2024-12-02T18:47:30+0000 lvl=eror msg=\"terminating with error\" obj=app err=\"authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n\"\n",
            "CRITICAL:pyngrok.process.ngrok:t=2024-12-02T18:47:30+0000 lvl=crit msg=\"command failed\" err=\"authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n\"\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "PyngrokNgrokError",
          "evalue": "The ngrok process errored on start: authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n.",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mPyngrokNgrokError\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-22-39a83e3d455b>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mpyngrok\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mngrok\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mpublic_url\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mngrok\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconnect\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m8501\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"Public URL: {public_url}\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyngrok/ngrok.py\u001b[0m in \u001b[0;36mconnect\u001b[0;34m(addr, proto, name, pyngrok_config, **options)\u001b[0m\n\u001b[1;32m    314\u001b[0m             \u001b[0moptions\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"auth\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    315\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 316\u001b[0;31m     \u001b[0mapi_url\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mget_ngrok_process\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mapi_url\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    317\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    318\u001b[0m     \u001b[0mlogger\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdebug\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"Creating tunnel with options: {options}\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyngrok/ngrok.py\u001b[0m in \u001b[0;36mget_ngrok_process\u001b[0;34m(pyngrok_config)\u001b[0m\n\u001b[1;32m    154\u001b[0m     \u001b[0minstall_ngrok\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    155\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 156\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mprocess\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_process\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    157\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    158\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyngrok/process.py\u001b[0m in \u001b[0;36mget_process\u001b[0;34m(pyngrok_config)\u001b[0m\n\u001b[1;32m    233\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0m_current_processes\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mngrok_path\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 235\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0m_start_process\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    236\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    237\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyngrok/process.py\u001b[0m in \u001b[0;36m_start_process\u001b[0;34m(pyngrok_config)\u001b[0m\n\u001b[1;32m    396\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    397\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mngrok_process\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstartup_error\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 398\u001b[0;31m             raise PyngrokNgrokError(f\"The ngrok process errored on start: {ngrok_process.startup_error}.\",\n\u001b[0m\u001b[1;32m    399\u001b[0m                                     \u001b[0mngrok_process\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlogs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    400\u001b[0m                                     ngrok_process.startup_error)\n",
            "\u001b[0;31mPyngrokNgrokError\u001b[0m: The ngrok process errored on start: authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n."
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Install necessary libraries\n",
        "!pip install streamlit pyngrok transformers\n",
        "\n",
        "# Save the Streamlit app code as app.py\n",
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "@st.cache_resource\n",
        "def load_model_and_tokenizer():\n",
        "    model = AutoModelForCausalLM.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    tokenizer = AutoTokenizer.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    return model, tokenizer\n",
        "\n",
        "model, tokenizer = load_model_and_tokenizer()\n",
        "special_token = \"<|endoftext|>\"\n",
        "\n",
        "st.title(\"Mental Wellbeing Chatbot\")\n",
        "user_input = st.text_input(\"You:\", placeholder=\"Type your message here...\")\n",
        "\n",
        "if st.button(\"Generate Response\"):\n",
        "    if user_input:\n",
        "        input_ids = tokenizer.encode(user_input + special_token, return_tensors=\"pt\")\n",
        "        output = model.generate(input_ids, max_length=100, num_return_sequences=1, top_p=0.9, temperature=0.7)\n",
        "        response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "        st.write(f\"Chatbot: {response}\")\n",
        "    else:\n",
        "        st.write(\"Please enter a message!\")\n",
        "\n",
        "# Set up and run ngrok to expose the Streamlit app\n",
        "from pyngrok import ngrok\n",
        "!streamlit run app.py &  # Start Streamlit in the background\n",
        "public_url = ngrok.connect(8501)  # Expose Streamlit app\n",
        "print(f\"Streamlit app is available at: {public_url}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rgNB0k9ys-Vy",
        "outputId": "1e082068-7f20-4635-9890-7398dc38f7df"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.40.2)\n",
            "Requirement already satisfied: pyngrok in /usr/local/lib/python3.10/dist-packages (7.2.1)\n",
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.46.2)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (11.0.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.25.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (17.0.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.9.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.43)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.10/dist-packages (from pyngrok) (6.0.2)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.16.1)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.26.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2024.9.11)\n",
            "Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.5)\n",
            "Requirement already satisfied: tokenizers<0.21,>=0.20 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.6)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (2024.9.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.8.30)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.18.0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.21.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.16.0)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "UsageError: Line magic function `%%writefile` not found.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Save the Streamlit app code as app.py\n",
        "\n",
        "app_code = \"\"\"\n",
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "@st.cache_resource\n",
        "def load_model_and_tokenizer():\n",
        "    model = AutoModelForCausalLM.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    tokenizer = AutoTokenizer.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    return model, tokenizer\n",
        "\n",
        "model, tokenizer = load_model_and_tokenizer()\n",
        "special_token = \"<|endoftext|>\"\n",
        "\n",
        "st.title(\"Mental Wellbeing Chatbot\")\n",
        "user_input = st.text_input(\"You:\", placeholder=\"Type your message here...\")\n",
        "\n",
        "if st.button(\"Generate Response\"):\n",
        "    if user_input:\n",
        "        input_ids = tokenizer.encode(user_input + special_token, return_tensors=\"pt\")\n",
        "        output = model.generate(input_ids, max_length=100, num_return_sequences=1, top_p=0.9, temperature=0.7)\n",
        "        response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "        st.write(f\"Chatbot: {response}\")\n",
        "    else:\n",
        "        st.write(\"Please enter a message!\")\n",
        "\"\"\"\n",
        "\n",
        "# Write the above code to the app.py file\n",
        "with open('/content/app.py', 'w') as file:\n",
        "    file.write(app_code)\n",
        "\n",
        "print(\"Streamlit app saved as app.py\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UsYsGicetVsV",
        "outputId": "0db4bc34-2a2d-4917-973e-462511890baa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Streamlit app saved as app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from pyngrok import ngrok\n",
        "\n",
        "# Start Streamlit in the background\n",
        "!streamlit run /content/app.py &\n",
        "\n",
        "# Expose the Streamlit app via ngrok\n",
        "public_url = ngrok.connect(8501)\n",
        "print(f\"Streamlit App is available at: {public_url}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 683
        },
        "id": "MDV2NX9Etbh4",
        "outputId": "e7331567-a45a-4f7a-ec5c-932796eb07ce"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://35.204.187.47:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "ERROR:pyngrok.process.ngrok:t=2024-12-02T19:02:14+0000 lvl=eror msg=\"failed to reconnect session\" obj=tunnels.session err=\"authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n\"\n",
            "ERROR:pyngrok.process.ngrok:t=2024-12-02T19:02:14+0000 lvl=eror msg=\"session closing\" obj=tunnels.session err=\"authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n\"\n",
            "ERROR:pyngrok.process.ngrok:t=2024-12-02T19:02:14+0000 lvl=eror msg=\"terminating with error\" obj=app err=\"authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n\"\n",
            "CRITICAL:pyngrok.process.ngrok:t=2024-12-02T19:02:14+0000 lvl=crit msg=\"command failed\" err=\"authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n\"\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "PyngrokNgrokError",
          "evalue": "The ngrok process errored on start: authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n.",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mPyngrokNgrokError\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-25-a68ff70a4de1>\u001b[0m in \u001b[0;36m<cell line: 7>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0;31m# Expose the Streamlit app via ngrok\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 7\u001b[0;31m \u001b[0mpublic_url\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mngrok\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconnect\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m8501\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      8\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"Streamlit App is available at: {public_url}\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyngrok/ngrok.py\u001b[0m in \u001b[0;36mconnect\u001b[0;34m(addr, proto, name, pyngrok_config, **options)\u001b[0m\n\u001b[1;32m    314\u001b[0m             \u001b[0moptions\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"auth\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    315\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 316\u001b[0;31m     \u001b[0mapi_url\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mget_ngrok_process\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mapi_url\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    317\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    318\u001b[0m     \u001b[0mlogger\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdebug\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"Creating tunnel with options: {options}\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyngrok/ngrok.py\u001b[0m in \u001b[0;36mget_ngrok_process\u001b[0;34m(pyngrok_config)\u001b[0m\n\u001b[1;32m    154\u001b[0m     \u001b[0minstall_ngrok\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    155\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 156\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mprocess\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_process\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    157\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    158\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyngrok/process.py\u001b[0m in \u001b[0;36mget_process\u001b[0;34m(pyngrok_config)\u001b[0m\n\u001b[1;32m    233\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0m_current_processes\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mngrok_path\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 235\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0m_start_process\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpyngrok_config\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    236\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    237\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pyngrok/process.py\u001b[0m in \u001b[0;36m_start_process\u001b[0;34m(pyngrok_config)\u001b[0m\n\u001b[1;32m    396\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    397\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mngrok_process\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstartup_error\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 398\u001b[0;31m             raise PyngrokNgrokError(f\"The ngrok process errored on start: {ngrok_process.startup_error}.\",\n\u001b[0m\u001b[1;32m    399\u001b[0m                                     \u001b[0mngrok_process\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlogs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    400\u001b[0m                                     ngrok_process.startup_error)\n",
            "\u001b[0;31mPyngrokNgrokError\u001b[0m: The ngrok process errored on start: authentication failed: The authtoken you specified does not look like a proper ngrok tunnel authtoken.\\nYour authtoken: YOUR_AUTHTOKEN\\nInstructions to install your authtoken are on your ngrok dashboard:\\nhttps://dashboard.ngrok.com/get-started/your-authtoken\\r\\n\\r\\nERR_NGROK_105\\r\\n."
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iZvvppjfwJYl",
        "outputId": "3fcb61f6-e54d-441f-a21e-33faaa2453e3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://35.204.187.47:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u74NhwBqypMd",
        "outputId": "bbf6dbcc-7c26-4038-bb8d-84e500f86026"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://35.204.187.47:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import json\n",
        "import random\n",
        "\n",
        "# Define a base dataset structure\n",
        "data = []\n",
        "\n",
        "# Common topics and queries\n",
        "queries = [\n",
        "    \"How do I manage my time better?\",\n",
        "    \"I feel anxious about exams.\",\n",
        "    \"How can I stay motivated to study?\",\n",
        "    \"What’s the best way to prepare for science?\",\n",
        "    \"Can you give me some study tips?\",\n",
        "    \"How do I deal with procrastination?\",\n",
        "    \"What’s the best way to revise for exams?\",\n",
        "    \"How should I manage multiple subjects?\",\n",
        "    \"I’m feeling overwhelmed. What should I do?\",\n",
        "    \"How can I focus better while studying?\",\n",
        "    \"What’s the Pomodoro technique?\",\n",
        "]\n",
        "\n",
        "# Responses to common queries\n",
        "responses = [\n",
        "    \"Create a schedule, prioritize tasks, and take regular breaks.\",\n",
        "    \"It's normal to feel anxious. Try deep breathing and mindfulness exercises.\",\n",
        "    \"Focus on the end goal and reward yourself for completing small tasks.\",\n",
        "    \"Break the subject into topics and allocate specific time for each topic.\",\n",
        "    \"Revise actively by summarizing topics and quizzing yourself.\",\n",
        "    \"Break tasks into smaller, manageable chunks and start with one small step.\",\n",
        "    \"Review your notes, create summaries, and test yourself with past papers.\",\n",
        "    \"Allocate time slots for each subject and stick to your plan.\",\n",
        "    \"Take a deep breath, list your priorities, and tackle them one at a time.\",\n",
        "    \"Minimize distractions, take breaks, and study in a quiet place.\",\n",
        "    \"Work for 25 minutes, then take a 5-minute break. Repeat for 4 cycles, then rest for 15 minutes.\",\n",
        "]\n",
        "\n",
        "# Fallback responses for ambiguous or unclear queries\n",
        "fallbacks = [\n",
        "    \"Can you clarify your question?\",\n",
        "    \"Could you provide more details so I can help?\",\n",
        "    \"I'm not sure I understand. Could you ask differently?\",\n",
        "    \"Let's focus on one thing at a time. What do you need help with first?\",\n",
        "    \"That's interesting! Can you tell me more about what you're asking?\",\n",
        "]\n",
        "\n",
        "# Generate 1000 examples with some fallback cases\n",
        "for i in range(1000):\n",
        "    if i % 10 == 0:  # Every 10th example is a fallback\n",
        "        # Select a random fallback response from the list\n",
        "        fallback = random.choice(fallbacks)\n",
        "        data.append({\"user\": \"What else?\", \"bot\": fallback})\n",
        "    elif i % 15 == 0:  # Edge case for \"and?\" or similar\n",
        "        data.append({\"user\": \"and?\", \"bot\": \"Can you be more specific? What else would you like to know?\"})\n",
        "    else:  # Regular responses\n",
        "        idx = i % len(queries)  # Cycle through queries and responses\n",
        "        data.append({\"user\": queries[idx], \"bot\": responses[idx]})\n",
        "\n",
        "# Save the dataset to a JSON file\n",
        "file_path = \"study_wellness_chatbot_dataset_1000.json\"\n",
        "with open(file_path, \"w\") as f:\n",
        "    json.dump(data, f, indent=2)\n",
        "\n",
        "print(f\"Dataset saved to {file_path}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UPLlZJyU61Nv",
        "outputId": "6217d07c-0948-4f7e-ab03-dc5e21a52fe2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Dataset saved to study_wellness_chatbot_dataset_1000.json\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install transformers datasets\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2S_6DHFJ7ZfY",
        "outputId": "50279a64-25cb-4a13-f987-7dbedfaad5ac"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.46.2)\n",
            "Collecting datasets\n",
            "  Downloading datasets-3.1.0-py3-none-any.whl.metadata (20 kB)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.16.1)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.26.2)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.26.4)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.2)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2024.9.11)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.32.3)\n",
            "Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.5)\n",
            "Requirement already satisfied: tokenizers<0.21,>=0.20 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.6)\n",
            "Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (17.0.0)\n",
            "Collecting dill<0.3.9,>=0.3.0 (from datasets)\n",
            "  Downloading dill-0.3.8-py3-none-any.whl.metadata (10 kB)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (2.2.2)\n",
            "Collecting xxhash (from datasets)\n",
            "  Downloading xxhash-3.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)\n",
            "Collecting multiprocess<0.70.17 (from datasets)\n",
            "  Downloading multiprocess-0.70.16-py310-none-any.whl.metadata (7.2 kB)\n",
            "Collecting fsspec<=2024.9.0,>=2023.1.0 (from fsspec[http]<=2024.9.0,>=2023.1.0->datasets)\n",
            "  Downloading fsspec-2024.9.0-py3-none-any.whl.metadata (11 kB)\n",
            "Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.11.2)\n",
            "Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (2.4.3)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)\n",
            "Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (24.2.0)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.5.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.1.0)\n",
            "Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (0.2.0)\n",
            "Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.17.2)\n",
            "Requirement already satisfied: async-timeout<6.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (4.12.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.8.30)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2024.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)\n",
            "Downloading datasets-3.1.0-py3-none-any.whl (480 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m480.6/480.6 kB\u001b[0m \u001b[31m27.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading dill-0.3.8-py3-none-any.whl (116 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m116.3/116.3 kB\u001b[0m \u001b[31m10.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading fsspec-2024.9.0-py3-none-any.whl (179 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m179.3/179.3 kB\u001b[0m \u001b[31m14.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading multiprocess-0.70.16-py310-none-any.whl (134 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m134.8/134.8 kB\u001b[0m \u001b[31m9.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading xxhash-3.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (194 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m194.1/194.1 kB\u001b[0m \u001b[31m15.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: xxhash, fsspec, dill, multiprocess, datasets\n",
            "  Attempting uninstall: fsspec\n",
            "    Found existing installation: fsspec 2024.10.0\n",
            "    Uninstalling fsspec-2024.10.0:\n",
            "      Successfully uninstalled fsspec-2024.10.0\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "gcsfs 2024.10.0 requires fsspec==2024.10.0, but you have fsspec 2024.9.0 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed datasets-3.1.0 dill-0.3.8 fsspec-2024.9.0 multiprocess-0.70.16 xxhash-3.5.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import datasets\n",
        "import transformers\n"
      ],
      "metadata": {
        "id": "Gn6V4ebk7eV0"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "per_device_train_batch_size=4\n",
        "per_device_eval_batch_size=4\n",
        "gradient_accumulation_steps=4  # To simulate a larger effective batch size\n"
      ],
      "metadata": {
        "id": "nz_YfAFF7iPW"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import json\n",
        "from transformers import (\n",
        "    GPT2LMHeadModel,\n",
        "    GPT2Tokenizer,\n",
        "    Trainer,\n",
        "    TrainingArguments,\n",
        ")\n",
        "from datasets import Dataset\n",
        "\n",
        "# Load GPT-2 model and tokenizer (use 'distilgpt2' for faster training if needed)\n",
        "model_name = \"gpt2\"  # Change to \"distilgpt2\" for smaller model\n",
        "model = GPT2LMHeadModel.from_pretrained(model_name)\n",
        "tokenizer = GPT2Tokenizer.from_pretrained(model_name)\n",
        "\n",
        "# Add a special token to separate user and bot messages\n",
        "special_token = \"<|endoftext|>\"\n",
        "tokenizer.add_special_tokens({\"pad_token\": special_token})\n",
        "model.resize_token_embeddings(len(tokenizer))\n",
        "\n",
        "# Step 1: Load Dataset\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Correct the file path\n",
        "file_path = \"/content/drive/My Drive/study_wellness_chatbot_dataset_1000.json\"\n",
        "with open(file_path, \"r\") as f:\n",
        "    data = json.load(f)\n",
        "\n",
        "\n",
        "# Prepare formatted data (User and Bot conversation)\n",
        "formatted_data = [\n",
        "    {\n",
        "        \"text\": f\"User: {example['user']}\\nBot: {example['bot']}{special_token}\"\n",
        "    }\n",
        "    for example in data\n",
        "]\n",
        "\n",
        "# Step 2: Convert to Hugging Face Dataset\n",
        "dataset = Dataset.from_dict({\"text\": [item[\"text\"] for item in formatted_data]})\n",
        "\n",
        "# Step 3: Tokenize Dataset\n",
        "def tokenize_function(example):\n",
        "    tokens = tokenizer(\n",
        "        example[\"text\"],\n",
        "        truncation=True,\n",
        "        padding=\"max_length\",\n",
        "        max_length=256,\n",
        "    )\n",
        "    tokens[\"labels\"] = tokens[\"input_ids\"].copy()  # Set labels for loss calculation\n",
        "    return tokens\n",
        "\n",
        "tokenized_dataset = dataset.map(tokenize_function, batched=True)\n",
        "tokenized_dataset.set_format(type=\"torch\", columns=[\"input_ids\", \"attention_mask\", \"labels\"])\n",
        "\n",
        "# Step 4: Split Dataset\n",
        "train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)\n",
        "train_dataset = train_test_split[\"train\"]\n",
        "val_dataset = train_test_split[\"test\"]\n",
        "\n",
        "print(f\"Training Data Size: {len(train_dataset)}\")\n",
        "print(f\"Validation Data Size: {len(val_dataset)}\")\n",
        "\n",
        "# Step 5: Training Arguments\n",
        "training_args = TrainingArguments(\n",
        "    output_dir=\"./fine_tuned_gpt2\",\n",
        "    evaluation_strategy=\"epoch\",  # Evaluate at the end of each epoch\n",
        "    logging_dir=\"./logs\",         # Log directory\n",
        "    logging_steps=200,            # Log every 200 steps\n",
        "    save_strategy=\"epoch\",        # Save at the end of each epoch\n",
        "    learning_rate=5e-5,           # Adjust learning rate for optimization\n",
        "    per_device_train_batch_size=8,  # Adjust based on GPU memory\n",
        "    per_device_eval_batch_size=8,\n",
        "    num_train_epochs=3,           # Set lower for faster testing, increase for full training\n",
        "    gradient_accumulation_steps=2,  # Simulates a larger batch size\n",
        "    weight_decay=0.01,            # Regularization\n",
        "    fp16=True,                    # Mixed precision for faster training\n",
        "    save_total_limit=2,           # Save only 2 checkpoints to save storage\n",
        "    report_to=\"none\",             # Disable reporting to external tools\n",
        ")\n",
        "\n",
        "# Step 6: Trainer Setup\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    train_dataset=train_dataset,\n",
        "    eval_dataset=val_dataset,\n",
        ")\n",
        "\n",
        "# Step 7: Fine-Tune the Model\n",
        "print(\"Starting training...\")\n",
        "trainer.train()\n",
        "\n",
        "# Step 8: Save Fine-Tuned Model\n",
        "model.save_pretrained(\"./fine_tuned_gpt2\")\n",
        "tokenizer.save_pretrained(\"./fine_tuned_gpt2\")\n",
        "print(\"Model fine-tuned and saved successfully!\")\n",
        "\n",
        "# Step 9: Test the Fine-Tuned Model\n",
        "from transformers import pipeline\n",
        "\n",
        "print(\"Testing the fine-tuned chatbot...\")\n",
        "chatbot = pipeline(\"text-generation\", model=\"./fine_tuned_gpt2\", tokenizer=tokenizer)\n",
        "\n",
        "while True:\n",
        "    user_input = input(\"You: \")\n",
        "    if user_input.lower() == \"exit\":\n",
        "        print(\"Chatbot: Goodbye!\")\n",
        "        break\n",
        "    response = chatbot(f\"User: {user_input}\\nBot:\", max_length=150, num_return_sequences=1)\n",
        "    print(\"Chatbot:\", response[0][\"generated_text\"].replace(\"User:\", \"\").replace(\"Bot:\", \"\").strip())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000,
          "referenced_widgets": [
            "55aaed8467354b0096a103ac4c6426a9",
            "c62ec2f5655540498fb3fd761bf0fa35",
            "a9d8dc65813f47ff9245c807381eea32",
            "ca7f75bcb18445c5882fa7bd465a39f4",
            "f89721670ba24234afa44af871ef8cb3",
            "020fa2e561484519b7aee29a70ed9204",
            "543025f54b0549eb9b1c1c165aa44127",
            "4a2f531898c445798ff1e316c79368a2",
            "4b1d37ed14e549c9ae07e7e83dc61d36",
            "e4a09d5fe9b0456f878669af845cf1ce",
            "367a1c07c02f4b01a03af2a2bdf1cdbc"
          ]
        },
        "id": "jsjpex3d8C7q",
        "outputId": "833b96f7-9c5a-4e97-df16-62f860cd2f8a"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "Map:   0%|          | 0/1000 [00:00<?, ? examples/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "55aaed8467354b0096a103ac4c6426a9"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training Data Size: 800\n",
            "Validation Data Size: 200\n",
            "Starting training...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/transformers/training_args.py:1568: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "    <div>\n",
              "      \n",
              "      <progress value='150' max='150' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
              "      [150/150 3:09:43, Epoch 3/3]\n",
              "    </div>\n",
              "    <table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              " <tr style=\"text-align: left;\">\n",
              "      <th>Epoch</th>\n",
              "      <th>Training Loss</th>\n",
              "      <th>Validation Loss</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <td>1</td>\n",
              "      <td>No log</td>\n",
              "      <td>0.042061</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <td>2</td>\n",
              "      <td>No log</td>\n",
              "      <td>0.014476</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <td>3</td>\n",
              "      <td>No log</td>\n",
              "      <td>0.013581</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table><p>"
            ]
          },
          "metadata": {}
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model fine-tuned and saved successfully!\n",
            "Testing the fine-tuned chatbot...\n",
            "You: Hi\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Chatbot: Hi\n",
            " I just wanted to try an experiment with mind mapping. What can I do? It's completely okay to be anxious. Try deep breathing or mindfulness exercises to calm your nerves.\n",
            "You: Time management\n",
            "Chatbot: Time management\n",
            " Start by creating a timer for each day to manage your time. Would you like help creating one?\n",
            "You: Yes\n",
            "Chatbot: Yes\n",
            " I just don’t feel like studying\n",
            " Sometimes just starting is the hardest part. Why not try studying for just 10 minutes and see how it feels?\n",
            "You: YEs\n",
            "Chatbot: YEs\n",
            " YEs are just the beginning of your project! Start by creating a study guide with notes and practice questions. Would you like help creating one?\n",
            "You: Yes\n",
            "Chatbot: Yes\n",
            " Deep breathing exercises and positive affirmations can help ease stress. Want to try a try one?\n",
            "You: Sure\n",
            "Chatbot: Sure\n",
            " You don't need to be perfect to make a difference. Find a way to stay motivated.\n",
            "You: I am bein anxieous\n",
            "Chatbot: I am bein anxieous\n",
            " Deep breathing exercises, stress management, or even a short walk can help ease anxiety. Want to try one?\n",
            "You: I want to try\n",
            "Chatbot: I want to try\n",
            " Have you tried active recall or spaced repetition?\n",
            "You: No\n",
            "Chatbot: No\n",
            " I’m not sure how to study for my exams, any advice?\n",
            " Use study techniques like mind mapping or flashcards to engage with the material actively. Let’s work on a plan together.\n",
            "You: Sure. Please explain\n",
            "Chatbot: Sure. Please explain\n",
            " I’m not sure how to study for my exams, any advice?\n",
            "You: I want to study for the exam\n",
            "Chatbot: I want to study for the exam\n",
            " Start by reviewing your notes and creating a study template with each subject. Would you like help creating one?\n",
            "You: Create a study plan\n",
            "Chatbot: Create a study plan\n",
            " Write down your goals and break them into manageable chunks. What’s one subject you can start with?\n",
            "You: Science\n",
            "Chatbot: Science\n",
            " Break your study sessions into focused blocks and make sure to take regular breaks. Want help creating a study schedule?\n",
            "You: Create study schedule\n",
            "Chatbot: Create study schedule\n",
            " Start with just 20 minutes and work your way through the material. Would you like help creating one?\n",
            "You: Create one\n",
            "Chatbot: Create one\n",
            " Start by creating a new Bot and creating a new password for each user. Would you like help creating one?\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "Interrupted by user",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-7-7df76278525b>\u001b[0m in \u001b[0;36m<cell line: 104>\u001b[0;34m()\u001b[0m\n\u001b[1;32m    103\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    104\u001b[0m \u001b[0;32mwhile\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 105\u001b[0;31m     \u001b[0muser_input\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"You: \"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    106\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0muser_input\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlower\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m\"exit\"\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    107\u001b[0m         \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Chatbot: Goodbye!\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/ipykernel/kernelbase.py\u001b[0m in \u001b[0;36mraw_input\u001b[0;34m(self, prompt)\u001b[0m\n\u001b[1;32m    849\u001b[0m                 \u001b[0;34m\"raw_input was called, but this frontend does not support input requests.\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    850\u001b[0m             )\n\u001b[0;32m--> 851\u001b[0;31m         return self._input_request(str(prompt),\n\u001b[0m\u001b[1;32m    852\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_parent_ident\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    853\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_parent_header\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/ipykernel/kernelbase.py\u001b[0m in \u001b[0;36m_input_request\u001b[0;34m(self, prompt, ident, parent, password)\u001b[0m\n\u001b[1;32m    893\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    894\u001b[0m                 \u001b[0;31m# re-raise KeyboardInterrupt, to truncate traceback\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 895\u001b[0;31m                 \u001b[0;32mraise\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Interrupted by user\"\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    896\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    897\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlog\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwarning\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Invalid Message:\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mexc_info\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: Interrupted by user"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "special_token = \"<|endoftext|>\"\n",
        "formatted_data = [\n",
        "    f\"User: {example['user']}\\nBot: {example['bot']}{special_token}\" for example in data\n",
        "]\n"
      ],
      "metadata": {
        "id": "hTQFSaicpTC9"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "# Load the fine-tuned model and tokenizer\n",
        "model = AutoModelForCausalLM.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "tokenizer = AutoTokenizer.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "special_token = \"<|endoftext|>\"\n",
        "\n",
        "# Set pad_token_id if not already set\n",
        "if tokenizer.pad_token_id is None:\n",
        "    tokenizer.pad_token_id = tokenizer.eos_token_id\n",
        "\n",
        "# Chat loop\n",
        "while True:\n",
        "    user_input = input(\"You: \")  # Get user input\n",
        "    if user_input.lower() == \"exit\":  # Exit condition\n",
        "        print(\"Chatbot: Goodbye!\")\n",
        "        break\n",
        "\n",
        "    # Tokenize user input\n",
        "    input_text = f\"User: {user_input}\\nBot:\"\n",
        "    input_ids = tokenizer.encode(input_text, return_tensors=\"pt\")  # Convert to tensor\n",
        "\n",
        "    # Create attention mask\n",
        "    attention_mask = input_ids.ne(tokenizer.pad_token_id)  # Identify valid tokens\n",
        "\n",
        "    # Generation configuration\n",
        "    generation_config = {\n",
        "        \"max_length\": len(input_ids[0]) + 100,  # Allow extra tokens for the response\n",
        "        \"num_return_sequences\": 1,             # Generate a single response\n",
        "        \"temperature\": 0.8,                    # Add randomness to the response\n",
        "        \"top_p\": 0.9,                          # Use nucleus sampling\n",
        "        \"do_sample\": True                      # Enable sampling-based generation\n",
        "    }\n",
        "\n",
        "    # Generate response\n",
        "    output = model.generate(\n",
        "        input_ids=input_ids,\n",
        "        attention_mask=attention_mask,\n",
        "        **generation_config\n",
        "    )\n",
        "\n",
        "    # Decode and format the response\n",
        "    response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "    bot_response = response.replace(input_text, \"\").strip()  # Remove user input from output\n",
        "    print(\"Chatbot:\", bot_response)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "twXXMtLvpYNJ",
        "outputId": "44c69c68-a016-4376-b2f1-ba67f2802836"
      },
      "execution_count": 10,
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "You: Hi\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: How do I stay motivated during finals?\n",
            "Bot: Focus on the end goal and break your study sessions into manageable chunks. What’s one subject you can start with?\n",
            "You: Time management \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Set clear priorities for each day and dedicate focused time blocks for studying. Do you want help planning your day?\n",
            "You: Please plan my day\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Start with a deep breathing exercise and stretch your body. Want to try one?\n",
            "You: Yes\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: I keep procrastinating, what can I do?\n",
            "You: Want to try\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Want help creating your own study guides? Want help creating one?\n",
            "You: Create one study guide\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Start by creating a study guide for each subject. Would you like help creating one?\n",
            "You: Create one\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Start by creating a new profile and creating a new account. You can start by creating a new profile and creating a new account.\n",
            "You: Okay, Thanks\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: I’m feeling anxious about my upcoming exams\n",
            "Bot: Deep breathing exercises and positive affirmations can help ease anxiety. Want to try one?\n",
            "You: Anxious \n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Start by setting a timer for 20 minutes and work on just one task. You might be surprised how much you can get done.\n",
            "You: I want to feel  motivated\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Start by setting small goals. Achieving even a little can motivate you to keep going.\n",
            "You: I am feeling anxious\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Deep breathing exercises and positive affirmations can help ease anxiety. Want to try one?\n",
            "You: I want to try exercises\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Take some slow, deep breaths or listen to calming music. Would you like to try a 5-minute guided meditation?\n",
            "You: 5 minutre guided meditation\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Deep breathing exercises and visualization exercises can help calm your mind. Let’s start with a deep breathing exercise.\n",
            "You: deep breathing exercise.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: deep breathing exercises can help you stay motivated. Want to try one?\n",
            "You: Want to try deep breathing exercise\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Want to try progressive muscle relaxation?\n",
            "You: Yes\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Use the Pomodoro technique—study for 25 minutes, then take a 5-minute break. It helps you stay focused.\n",
            "You: Okay\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: I just don’t feel like studying\n",
            "Bot: Sometimes just starting is the hardest part. Why not try studying for just 10 minutes and see how it feels?\n",
            "You: I want some study tips\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: Start by reviewing your notes and creating a study guide for each subject. Have you tried active recall or spaced repetition?\n",
            "You: No\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Chatbot: I have pre-exam stress, can you help?\n",
            "Bot: Yes, relaxation techniques can help. Try some slow, deep breaths and stretch your body.\n",
            "You: Anxiety\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:None for open-end generation.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Chatbot: It's okay. Try deep breathing or mindfulness exercises to calm your nerves.\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "Interrupted by user",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-10-4a4d5d43d679>\u001b[0m in \u001b[0;36m<cell line: 13>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     12\u001b[0m \u001b[0;31m# Chat loop\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m \u001b[0;32mwhile\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 14\u001b[0;31m     \u001b[0muser_input\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"You: \"\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# Get user input\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     15\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0muser_input\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlower\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m\"exit\"\u001b[0m\u001b[0;34m:\u001b[0m  \u001b[0;31m# Exit condition\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     16\u001b[0m         \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Chatbot: Goodbye!\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/ipykernel/kernelbase.py\u001b[0m in \u001b[0;36mraw_input\u001b[0;34m(self, prompt)\u001b[0m\n\u001b[1;32m    849\u001b[0m                 \u001b[0;34m\"raw_input was called, but this frontend does not support input requests.\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    850\u001b[0m             )\n\u001b[0;32m--> 851\u001b[0;31m         return self._input_request(str(prompt),\n\u001b[0m\u001b[1;32m    852\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_parent_ident\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    853\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_parent_header\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/ipykernel/kernelbase.py\u001b[0m in \u001b[0;36m_input_request\u001b[0;34m(self, prompt, ident, parent, password)\u001b[0m\n\u001b[1;32m    893\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    894\u001b[0m                 \u001b[0;31m# re-raise KeyboardInterrupt, to truncate traceback\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 895\u001b[0;31m                 \u001b[0;32mraise\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Interrupted by user\"\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    896\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    897\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlog\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mwarning\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Invalid Message:\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mexc_info\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: Interrupted by user"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# prompt: Using streamlit create a GUI interface to have this conversation with our finetuned model\n",
        "\n",
        "!pip install streamlit pyngrok transformers\n",
        "\n",
        "# Assuming your fine-tuned model and tokenizer are saved in the directory \"./fine_tuned_gpt2\"\n",
        "# Replace with your actual paths if necessary\n",
        "\n",
        "# Streamlit app code\n",
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "@st.cache_resource\n",
        "def load_model_and_tokenizer():\n",
        "    model = AutoModelForCausalLM.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    tokenizer = AutoTokenizer.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    return model, tokenizer\n",
        "\n",
        "model, tokenizer = load_model_and_tokenizer()\n",
        "special_token = \"<|endoftext|>\"\n",
        "\n",
        "st.title(\"Mental Wellbeing Chatbot\")\n",
        "user_input = st.text_input(\"You:\", placeholder=\"Type your message here...\")\n",
        "\n",
        "if st.button(\"Generate Response\"):\n",
        "    if user_input:\n",
        "        input_ids = tokenizer.encode(user_input + special_token, return_tensors=\"pt\")\n",
        "        output = model.generate(input_ids, max_length=100, num_return_sequences=1, top_p=0.9, temperature=0.7)\n",
        "        response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "        st.write(f\"Chatbot: {response}\")\n",
        "    else:\n",
        "        st.write(\"Please enter a message!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xX5TMInnrtCS",
        "outputId": "fa2c5657-febc-4bc5-cf39-6b6c968630c1"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.40.2)\n",
            "Requirement already satisfied: pyngrok in /usr/local/lib/python3.10/dist-packages (7.2.1)\n",
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.46.2)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (11.0.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.25.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (17.0.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.9.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.43)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.10/dist-packages (from pyngrok) (6.0.2)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.16.1)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.26.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2024.9.11)\n",
            "Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.5)\n",
            "Requirement already satisfied: tokenizers<0.21,>=0.20 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.6)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (2024.9.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.8.30)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.18.0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.21.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.16.0)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "UsageError: Line magic function `%%writefile` not found.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "# Load your fine-tuned model and tokenizer\n",
        "@st.cache_resource  # Cache the model and tokenizer for faster loading\n",
        "def load_model_and_tokenizer():\n",
        "    model = AutoModelForCausalLM.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    tokenizer = AutoTokenizer.from_pretrained(\"./fine_tuned_gpt2\")\n",
        "    return model, tokenizer\n",
        "\n",
        "model, tokenizer = load_model_and_tokenizer()\n",
        "special_token = \"<|endoftext|>\"  # Your special token\n",
        "\n",
        "# Streamlit app title\n",
        "st.title(\"Mental Wellbeing Chatbot\")\n",
        "\n",
        "# User input area\n",
        "user_input = st.text_input(\"You:\")\n",
        "\n",
        "# Generate response button\n",
        "if st.button(\"Generate Response\"):\n",
        "    if user_input:\n",
        "        # Generate response using the model\n",
        "        input_ids = tokenizer.encode(user_input + special_token, return_tensors=\"pt\")\n",
        "        output = model.generate(input_ids, max_length=100, num_return_sequences=1, top_p=0.9, temperature=0.7)  # Adjust parameters as needed\n",
        "        response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "\n",
        "        # Display the response\n",
        "        st.write(\"Chatbot:\", response)\n",
        "    else:\n",
        "        st.write(\"Please enter a message.\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "m2eVw0ZcsXCV",
        "outputId": "57e50212-6d71-40bc-8031-47af1db9022f"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NIvyCJXesagg",
        "outputId": "a5d4be37-25c4-4d9e-b645-ca5779e4422c"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.48.205.247:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "^C\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install streamlit transformers\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "As2qcRgdtY_x",
        "outputId": "1b431825-1952-42ae-8f42-76c401480120"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.40.2)\n",
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.46.2)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.2)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (11.0.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.25.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (17.0.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.9.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.0.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.43)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.16.1)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.26.2)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.2)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2024.9.11)\n",
            "Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.5)\n",
            "Requirement already satisfied: tokenizers<0.21,>=0.20 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.6)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (2024.9.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.4.0->streamlit) (2024.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.4.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.8.30)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.18.0)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2024.10.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.21.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.16.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!mkdir chatbot_app\n",
        "!cd chatbot_app\n"
      ],
      "metadata": {
        "id": "tCXqNSIIte3G"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "# Mount Google Drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Define the path to save the model in Google Drive\n",
        "model_save_path = '/content/drive/MyDrive/chatbot_model'\n",
        "\n",
        "# Load your fine-tuned model and tokenizer\n",
        "model = AutoModelForCausalLM.from_pretrained(\"gpt2\")  # Use your fine-tuned model here\n",
        "tokenizer = AutoTokenizer.from_pretrained(\"gpt2\")    # Use your fine-tuned tokenizer here\n",
        "\n",
        "# Save the model and tokenizer to Google Drive\n",
        "model.save_pretrained(model_save_path)\n",
        "tokenizer.save_pretrained(model_save_path)\n",
        "\n",
        "print(f\"Model and tokenizer saved to {model_save_path}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "P4Vdo_YkuiG3",
        "outputId": "01a85c1e-5f9c-45f5-8da7-826adef95ebc"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n",
            "Model and tokenizer saved to /content/drive/MyDrive/chatbot_model\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "# Mount Google Drive to access the saved model\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Path to your saved model in Google Drive\n",
        "model_path = '/content/drive/MyDrive/chatbot_model'\n",
        "\n",
        "# Load the fine-tuned model and tokenizer\n",
        "@st.cache_resource  # Cache the model for faster reloads\n",
        "def load_model_and_tokenizer():\n",
        "    model = AutoModelForCausalLM.from_pretrained(model_path)\n",
        "    tokenizer = AutoTokenizer.from_pretrained(model_path)\n",
        "    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if not defined\n",
        "    return model, tokenizer\n",
        "\n",
        "model, tokenizer = load_model_and_tokenizer()\n",
        "\n",
        "# Special token\n",
        "special_token = \"<|endoftext|>\"\n",
        "\n",
        "# App title\n",
        "st.title(\"Study & Wellbeing Chatbot\")\n",
        "\n",
        "# Sidebar for info\n",
        "st.sidebar.title(\"About\")\n",
        "st.sidebar.info(\"This chatbot helps you with study tips and managing anxiety during exams.\")\n",
        "\n",
        "# User input\n",
        "user_input = st.text_input(\"You:\", placeholder=\"Ask me anything about studies or stress relief!\")\n",
        "\n",
        "# Generate response when button is clicked\n",
        "if st.button(\"Get Response\"):\n",
        "    if user_input:\n",
        "        input_text = f\"User: {user_input}\\nBot:\"\n",
        "        input_ids = tokenizer.encode(input_text, return_tensors=\"pt\")\n",
        "\n",
        "        # Generate response\n",
        "        output = model.generate(\n",
        "            input_ids,\n",
        "            max_length=150,\n",
        "            num_return_sequences=1,\n",
        "            temperature=0.8,\n",
        "            top_p=0.9,\n",
        "            do_sample=True\n",
        "        )\n",
        "\n",
        "        # Decode and display the response\n",
        "        response = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "        bot_response = response.replace(input_text, \"\").strip()\n",
        "        st.write(f\"Chatbot: {bot_response}\")\n",
        "    else:\n",
        "        st.write(\"Please enter a message.\")\n",
        "\n",
        "# Footer\n",
        "st.sidebar.text(\"Developed with ❤️ using Streamlit and Transformers\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cNwYW5Mnwn_U",
        "outputId": "a647b454-b49d-4ca6-9239-75458092815a"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-12-03 04:26:01.039 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.217 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2024-12-03 04:26:01.223 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.226 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-12-03 04:26:01.762 Thread 'Thread-13': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.767 Thread 'Thread-13': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.780 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.782 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.786 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.788 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.790 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.791 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.792 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.794 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.799 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.800 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.804 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.807 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.809 Session state does not function when running a script without `streamlit run`\n",
            "2024-12-03 04:26:01.812 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.814 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.819 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.821 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.824 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.827 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.829 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.832 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-12-03 04:26:01.835 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator(_root_container=1, _parent=DeltaGenerator())"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0_cvDryAxNHn",
        "outputId": "76b9c0dd-9416-4183-fa62-39aa95122695"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.48.205.247:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git init\n",
        "!git add .\n",
        "!git commit -m \"Initial commit\"\n",
        "!git branch -M main\n",
        "!git remote add origin <https://github.com/rahusharma2312/pkl>\n",
        "!git push -u origin main\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WaJB8r6Vxnx2",
        "outputId": "c460af65-fbbb-4189-a9d3-27ad5bbdd1fa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[33mhint: Using 'master' as the name for the initial branch. This default branch name\u001b[m\n",
            "\u001b[33mhint: is subject to change. To configure the initial branch name to use in all\u001b[m\n",
            "\u001b[33mhint: of your new repositories, which will suppress this warning, call:\u001b[m\n",
            "\u001b[33mhint: \u001b[m\n",
            "\u001b[33mhint: \tgit config --global init.defaultBranch <name>\u001b[m\n",
            "\u001b[33mhint: \u001b[m\n",
            "\u001b[33mhint: Names commonly chosen instead of 'master' are 'main', 'trunk' and\u001b[m\n",
            "\u001b[33mhint: 'development'. The just-created branch can be renamed via this command:\u001b[m\n",
            "\u001b[33mhint: \u001b[m\n",
            "\u001b[33mhint: \tgit branch -m <name>\u001b[m\n",
            "Initialized empty Git repository in /content/.git/\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "IB_iZI0csWuN"
      }
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    },
    "widgets": {
      "application/vnd.jupyter.widget-state+json": {
        "022fef9b271446e99b078dc0cef4ec01": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "02ff813c0c5b4262b79c400515bee803": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "02ffb12291b347dc8eb7e24c045facd3": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "07a97e941f824b8a89eb06c51020732a": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_ffc10a5826bf4e419d8536bad3fccf0a",
            "placeholder": "​",
            "style": "IPY_MODEL_022fef9b271446e99b078dc0cef4ec01",
            "value": " 1000/1000 [00:00&lt;00:00, 24558.11 examples/s]"
          }
        },
        "11908c02fc594707ba856a0b19f54a39": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "16884993e4734bcabd7a44cd7e6a4068": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_84f49143155141189418c317de1e37a0",
            "placeholder": "​",
            "style": "IPY_MODEL_ab56c0c791864cedb30bc50c059501fd",
            "value": " 1000/1000 [00:00&lt;00:00, 1276.43 examples/s]"
          }
        },
        "180e3ed4bd514214905c9965709463a5": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "1cbad58c80fc47d68d5a6cffbf118ea0": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_aeca09d2255b4806bf23c119f4b79f5f",
              "IPY_MODEL_2261af50888f41ce91e1d3792782254f",
              "IPY_MODEL_cd113e7f134c4d2d9a0df0c092b149c4"
            ],
            "layout": "IPY_MODEL_dc9986714a164a91a1f82941371f7e28"
          }
        },
        "21966790e0b548748c799ede793edf01": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "21ab70a692a74ab3b61c1ea2c74ed3c5": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_56271191644245ef8647491effd47da2",
            "placeholder": "​",
            "style": "IPY_MODEL_f0094929c546446fa60146d868fafe41",
            "value": "Map: 100%"
          }
        },
        "2261af50888f41ce91e1d3792782254f": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_b225767c1ea44f98a11bb9b0e3473586",
            "max": 800,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_34c99f1b234640cdbfb8eb2bc3187511",
            "value": 800
          }
        },
        "25316aa8fd5d462696b5de856ba6ff82": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "274db7a30168456a821c0330c7a16d67": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_d241cfb7ddf24254be90d25eb51d35d7",
            "placeholder": "​",
            "style": "IPY_MODEL_44b411a900364a14bfffb50f8cdefaa8",
            "value": " 800/800 [00:00&lt;00:00, 2457.66 examples/s]"
          }
        },
        "2d28f0df9de1443ca21a4c2277988e33": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "3192dce3a9ae4f60b1b289d8170a0f00": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_e9754edfedb442b0979b9d734796c23c",
            "max": 1000,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_a44971847ef64d07bd611f76e609a1c3",
            "value": 1000
          }
        },
        "32bff537f2ef45af9913fe49521999b8": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_25316aa8fd5d462696b5de856ba6ff82",
            "max": 1000,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_b66b213bf23e4a968eb4653bc67ac258",
            "value": 1000
          }
        },
        "34abdb1f8bfa48939c621eea27e625ee": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_21ab70a692a74ab3b61c1ea2c74ed3c5",
              "IPY_MODEL_3192dce3a9ae4f60b1b289d8170a0f00",
              "IPY_MODEL_16884993e4734bcabd7a44cd7e6a4068"
            ],
            "layout": "IPY_MODEL_c5cefc4e110e4c09937bdf7c379317eb"
          }
        },
        "34c99f1b234640cdbfb8eb2bc3187511": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "35dd7fdf2dd944ee90a7ea055af49d3f": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_180e3ed4bd514214905c9965709463a5",
            "placeholder": "​",
            "style": "IPY_MODEL_c7fed0e6c44c485ca1efabe4d098a23a",
            "value": "Saving the dataset (1/1 shards): 100%"
          }
        },
        "38e6d6afc4a54816929c6b7c1c377ceb": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "3aae5fea0fbe4196996b9942db2f71db": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "3c26df8da94246d98f3f4e665aa578a5": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_b9a9a1975883475f912d39f953f99ef0",
            "max": 200,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_b37ae495a68143618967005297e86357",
            "value": 200
          }
        },
        "3c80aa281c194dc4a2cd1b93af6d74d2": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_35dd7fdf2dd944ee90a7ea055af49d3f",
              "IPY_MODEL_df52b3f366ea474a897fddd6bf7afc58",
              "IPY_MODEL_07a97e941f824b8a89eb06c51020732a"
            ],
            "layout": "IPY_MODEL_693ad5e21ad640b6ac29e6da994a5689"
          }
        },
        "3d269a01672b493aac5ddde9ad5ce697": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_e15c6783315549b997fc265d422421bd",
            "max": 1000,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_995c249c4def48c49c24555a8b8b045a",
            "value": 1000
          }
        },
        "400d431a85f34cc3aa3260d83b07f6c4": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "429bfb9f034d4e7f8f6dbc115ced478a": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "42b3e8391c4f4621864eea7e949420bc": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_8bd6d5e9bf11447caa007bd86003c5b1",
              "IPY_MODEL_3d269a01672b493aac5ddde9ad5ce697",
              "IPY_MODEL_b300f837f2da425bb9ecd0477676512a"
            ],
            "layout": "IPY_MODEL_aa7236e6297e4e3cbf7ffcd973b2aa8f"
          }
        },
        "44b411a900364a14bfffb50f8cdefaa8": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "452d0102516f470082411f25b6ea8145": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "55214a89e4a449f29cf98f67f2ce28bf": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_6daf03bb5b8d4cad9de207c89b8e159e",
            "placeholder": "​",
            "style": "IPY_MODEL_d1391f707a2241e9931d28a7a28aa8dc",
            "value": "Map: 100%"
          }
        },
        "56271191644245ef8647491effd47da2": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "58e00ef39a89465fb3fedabf3ff32f58": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "599321f00c9f481888f6e64945dc9d41": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "5a271d59a18c4260a0824f8943f7115b": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_8ae626901eb143a380e31e536442e646",
            "placeholder": "​",
            "style": "IPY_MODEL_2d28f0df9de1443ca21a4c2277988e33",
            "value": " 1000/1000 [00:00&lt;00:00, 16496.97 examples/s]"
          }
        },
        "62d3a148c4c04892bc112e03445d9cf3": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "650e7741c95a4111aae655e8e2bc321a": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "693ad5e21ad640b6ac29e6da994a5689": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "6daf03bb5b8d4cad9de207c89b8e159e": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "702205a3b42a4e31a858994cfe138b6b": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_599321f00c9f481888f6e64945dc9d41",
            "placeholder": "​",
            "style": "IPY_MODEL_7f77db9721154629af5a4a10e137bd5b",
            "value": " 200/200 [00:00&lt;00:00, 1694.42 examples/s]"
          }
        },
        "749c9e768c0f4a029ef3dbdcfb64b8ed": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_55214a89e4a449f29cf98f67f2ce28bf",
              "IPY_MODEL_3c26df8da94246d98f3f4e665aa578a5",
              "IPY_MODEL_b40e82b7f9054201bdc8be5bd0f4626c"
            ],
            "layout": "IPY_MODEL_02ffb12291b347dc8eb7e24c045facd3"
          }
        },
        "7f77db9721154629af5a4a10e137bd5b": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "834e5897c6374a49a3e15b07c13f9a65": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_21966790e0b548748c799ede793edf01",
            "placeholder": "​",
            "style": "IPY_MODEL_8f2fec4605b84cd5aa7320a865b103ea",
            "value": "Saving the dataset (1/1 shards): 100%"
          }
        },
        "83e10f4831c545b79414094a4bbe6fcd": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_d087616599954d7ea653f17a358edba8",
              "IPY_MODEL_f895362ff17840e6a85fcf220e5aab6d",
              "IPY_MODEL_274db7a30168456a821c0330c7a16d67"
            ],
            "layout": "IPY_MODEL_e165670cd963463d90c56a3b056e12dc"
          }
        },
        "84f49143155141189418c317de1e37a0": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "8ae626901eb143a380e31e536442e646": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "8bd6d5e9bf11447caa007bd86003c5b1": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_8fbdfe7f05394ddc831e856a1593942c",
            "placeholder": "​",
            "style": "IPY_MODEL_a7b6c066853d4756824d7f8dabf37409",
            "value": "Map: 100%"
          }
        },
        "8f2fec4605b84cd5aa7320a865b103ea": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "8fbdfe7f05394ddc831e856a1593942c": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "97a7d6c4e07e40c1b060f87a31d07438": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "995c249c4def48c49c24555a8b8b045a": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "a44971847ef64d07bd611f76e609a1c3": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "a7b6c066853d4756824d7f8dabf37409": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "aa7236e6297e4e3cbf7ffcd973b2aa8f": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "ab56c0c791864cedb30bc50c059501fd": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "ab8a8de99e7f4cdc8b5bafcc43c91626": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_97a7d6c4e07e40c1b060f87a31d07438",
            "max": 200,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_3aae5fea0fbe4196996b9942db2f71db",
            "value": 200
          }
        },
        "aebd0a2a384b42fd96b0b532e305a7db": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_58e00ef39a89465fb3fedabf3ff32f58",
            "placeholder": "​",
            "style": "IPY_MODEL_11908c02fc594707ba856a0b19f54a39",
            "value": "Map: 100%"
          }
        },
        "aeca09d2255b4806bf23c119f4b79f5f": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_452d0102516f470082411f25b6ea8145",
            "placeholder": "​",
            "style": "IPY_MODEL_cd1bec76be0e48d39f0728172f8ae769",
            "value": "Map: 100%"
          }
        },
        "b225767c1ea44f98a11bb9b0e3473586": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "b2c653b79f1f486eb461424e6ec9ddb6": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "b300f837f2da425bb9ecd0477676512a": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_38e6d6afc4a54816929c6b7c1c377ceb",
            "placeholder": "​",
            "style": "IPY_MODEL_650e7741c95a4111aae655e8e2bc321a",
            "value": " 1000/1000 [00:00&lt;00:00, 1115.97 examples/s]"
          }
        },
        "b37ae495a68143618967005297e86357": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "b40e82b7f9054201bdc8be5bd0f4626c": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_f76d4a1bbdb94d44bfa09375d0bbd099",
            "placeholder": "​",
            "style": "IPY_MODEL_b2c653b79f1f486eb461424e6ec9ddb6",
            "value": " 200/200 [00:00&lt;00:00, 1141.79 examples/s]"
          }
        },
        "b64792673e4b45908935df2618f29763": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "b66b213bf23e4a968eb4653bc67ac258": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "b9a9a1975883475f912d39f953f99ef0": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "c5cefc4e110e4c09937bdf7c379317eb": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "c7fed0e6c44c485ca1efabe4d098a23a": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "ca122f4897b047669638fc9dcbf7fb13": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_834e5897c6374a49a3e15b07c13f9a65",
              "IPY_MODEL_32bff537f2ef45af9913fe49521999b8",
              "IPY_MODEL_5a271d59a18c4260a0824f8943f7115b"
            ],
            "layout": "IPY_MODEL_02ff813c0c5b4262b79c400515bee803"
          }
        },
        "cd113e7f134c4d2d9a0df0c092b149c4": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_429bfb9f034d4e7f8f6dbc115ced478a",
            "placeholder": "​",
            "style": "IPY_MODEL_400d431a85f34cc3aa3260d83b07f6c4",
            "value": " 800/800 [00:00&lt;00:00, 2001.34 examples/s]"
          }
        },
        "cd1bec76be0e48d39f0728172f8ae769": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "d087616599954d7ea653f17a358edba8": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_fc3d94242522474a94dbb2d90a0ed22f",
            "placeholder": "​",
            "style": "IPY_MODEL_d3d80fcf0cbc491ca3d8dd7f9ef51ecf",
            "value": "Map: 100%"
          }
        },
        "d1391f707a2241e9931d28a7a28aa8dc": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "d241cfb7ddf24254be90d25eb51d35d7": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "d3d80fcf0cbc491ca3d8dd7f9ef51ecf": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "d88461475dab493aa3c2c1f08ed4b95f": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "dc9986714a164a91a1f82941371f7e28": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "df52b3f366ea474a897fddd6bf7afc58": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_d88461475dab493aa3c2c1f08ed4b95f",
            "max": 1000,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_e06bd5fae4654b36ac9451030988e9eb",
            "value": 1000
          }
        },
        "e06bd5fae4654b36ac9451030988e9eb": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "e15c6783315549b997fc265d422421bd": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "e165670cd963463d90c56a3b056e12dc": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "e26e39930f1942aab183ce7578330784": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "VBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "VBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "VBoxView",
            "box_style": "",
            "children": [],
            "layout": "IPY_MODEL_e270679ad8fe4407a62e50e671e95fc0"
          }
        },
        "e270679ad8fe4407a62e50e671e95fc0": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": "center",
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": "flex",
            "flex": null,
            "flex_flow": "column",
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": "50%"
          }
        },
        "e9754edfedb442b0979b9d734796c23c": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "ed5b2edeb79549789cd4aac2a95a6a03": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "f0094929c546446fa60146d868fafe41": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "f090ab4c142845faad2fb0ae44304160": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_aebd0a2a384b42fd96b0b532e305a7db",
              "IPY_MODEL_ab8a8de99e7f4cdc8b5bafcc43c91626",
              "IPY_MODEL_702205a3b42a4e31a858994cfe138b6b"
            ],
            "layout": "IPY_MODEL_b64792673e4b45908935df2618f29763"
          }
        },
        "f76d4a1bbdb94d44bfa09375d0bbd099": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "f895362ff17840e6a85fcf220e5aab6d": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_ed5b2edeb79549789cd4aac2a95a6a03",
            "max": 800,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_62d3a148c4c04892bc112e03445d9cf3",
            "value": 800
          }
        },
        "fc3d94242522474a94dbb2d90a0ed22f": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "ffc10a5826bf4e419d8536bad3fccf0a": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "b434bd1b0f7c43298880e9ee99157082": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_4655a50836744364b3aa9cf8effb1972",
              "IPY_MODEL_4eb78fe04dc046e78cbfccaac7c3f015",
              "IPY_MODEL_87cb6cc946564673a9848bd5ee69033c"
            ],
            "layout": "IPY_MODEL_fde47086348640fd86d43757f8de5af5"
          }
        },
        "4655a50836744364b3aa9cf8effb1972": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_5b7e0ad9c40d43abb52bb592e997b197",
            "placeholder": "​",
            "style": "IPY_MODEL_86e38b472d7c4fababbae0b3d5e267cb",
            "value": "config.json: 100%"
          }
        },
        "4eb78fe04dc046e78cbfccaac7c3f015": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_68773fad9d6e40569f4f6955e6a1b437",
            "max": 665,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_0fa3656a16aa4899ae387e8bb24cfb49",
            "value": 665
          }
        },
        "87cb6cc946564673a9848bd5ee69033c": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_2cad87eee216405bbd8dfec379568a9b",
            "placeholder": "​",
            "style": "IPY_MODEL_e51cd84ee298440e8f87e163fb0b4bd8",
            "value": " 665/665 [00:00&lt;00:00, 39.4kB/s]"
          }
        },
        "fde47086348640fd86d43757f8de5af5": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "5b7e0ad9c40d43abb52bb592e997b197": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "86e38b472d7c4fababbae0b3d5e267cb": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "68773fad9d6e40569f4f6955e6a1b437": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "0fa3656a16aa4899ae387e8bb24cfb49": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "2cad87eee216405bbd8dfec379568a9b": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "e51cd84ee298440e8f87e163fb0b4bd8": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "ccbb174dc1ed46b6a9f1784e52120284": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_47f8aaf3ba6244d8bb90cc6678c32a30",
              "IPY_MODEL_8001fee05cab415daa8ae4a4562dd2f3",
              "IPY_MODEL_35da76c1c0c24b06aa84d1f6b83fd782"
            ],
            "layout": "IPY_MODEL_4e97eba606eb4f038334c9f08ee08f2b"
          }
        },
        "47f8aaf3ba6244d8bb90cc6678c32a30": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_bdd80cb862884145a3cfc685b3480310",
            "placeholder": "​",
            "style": "IPY_MODEL_e9f79b49dabc45a8a17d09dc4d3379c4",
            "value": "model.safetensors: 100%"
          }
        },
        "8001fee05cab415daa8ae4a4562dd2f3": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_b7f5ca8f76ef4d16b48bb3f3d6c44604",
            "max": 548105171,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_34f11e67713748b8b26d5be52d457c26",
            "value": 548105171
          }
        },
        "35da76c1c0c24b06aa84d1f6b83fd782": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_dcaa562147714889944da555eef5a39a",
            "placeholder": "​",
            "style": "IPY_MODEL_58c1c57bdc234b66854934b9f6245db2",
            "value": " 548M/548M [00:02&lt;00:00, 244MB/s]"
          }
        },
        "4e97eba606eb4f038334c9f08ee08f2b": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "bdd80cb862884145a3cfc685b3480310": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "e9f79b49dabc45a8a17d09dc4d3379c4": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "b7f5ca8f76ef4d16b48bb3f3d6c44604": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "34f11e67713748b8b26d5be52d457c26": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "dcaa562147714889944da555eef5a39a": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "58c1c57bdc234b66854934b9f6245db2": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "5f5bff86f6354e5b8536f71519696c55": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_a59a6206590940cbbfe72fc56c2ce20a",
              "IPY_MODEL_a027a7a380274c2b8201fe9529cb00a0",
              "IPY_MODEL_9122b6cd216c4669a20c9d9b6fbd3e01"
            ],
            "layout": "IPY_MODEL_a12b1d0922514901b35f1abfcb28fe0a"
          }
        },
        "a59a6206590940cbbfe72fc56c2ce20a": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_22d9e9afc8c14eb18917279c4768ebf3",
            "placeholder": "​",
            "style": "IPY_MODEL_4d2ba261f8784e27b08402004e6b4d17",
            "value": "generation_config.json: 100%"
          }
        },
        "a027a7a380274c2b8201fe9529cb00a0": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_6a3e25ebb8e9428999018d5346f0312c",
            "max": 124,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_c20831b89da24fef985c65d16d8314c9",
            "value": 124
          }
        },
        "9122b6cd216c4669a20c9d9b6fbd3e01": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_4e64bca526e1457c91f3d78d31a28c0e",
            "placeholder": "​",
            "style": "IPY_MODEL_477a5ad7c3894cc690d263e09d4485dd",
            "value": " 124/124 [00:00&lt;00:00, 6.03kB/s]"
          }
        },
        "a12b1d0922514901b35f1abfcb28fe0a": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "22d9e9afc8c14eb18917279c4768ebf3": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "4d2ba261f8784e27b08402004e6b4d17": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "6a3e25ebb8e9428999018d5346f0312c": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "c20831b89da24fef985c65d16d8314c9": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "4e64bca526e1457c91f3d78d31a28c0e": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "477a5ad7c3894cc690d263e09d4485dd": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "d2279efe79084bafba625c3733a1d825": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_5a757873adad4feba890b0d3b01c76e1",
              "IPY_MODEL_ba5940168cbf4badb8e1aea2adb7965e",
              "IPY_MODEL_9966803aa4f8434284df70dc65d1df93"
            ],
            "layout": "IPY_MODEL_715a7ed221b84970a76ec01e9b8c2251"
          }
        },
        "5a757873adad4feba890b0d3b01c76e1": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_d414bbc106ca4c10b8ebd5e3b96421da",
            "placeholder": "​",
            "style": "IPY_MODEL_7cf2adc7aa5744e7b92dacbd05707797",
            "value": "tokenizer_config.json: 100%"
          }
        },
        "ba5940168cbf4badb8e1aea2adb7965e": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_4f970a7bb13f461084c9a19335300961",
            "max": 26,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_5f8fb54f9c6f49aa83c8ebae33844059",
            "value": 26
          }
        },
        "9966803aa4f8434284df70dc65d1df93": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_9cdcf372239b405d8ade6ba8f836ff9a",
            "placeholder": "​",
            "style": "IPY_MODEL_81d4531bc53646c0837de049e7e5fbf5",
            "value": " 26.0/26.0 [00:00&lt;00:00, 1.16kB/s]"
          }
        },
        "715a7ed221b84970a76ec01e9b8c2251": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "d414bbc106ca4c10b8ebd5e3b96421da": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "7cf2adc7aa5744e7b92dacbd05707797": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "4f970a7bb13f461084c9a19335300961": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "5f8fb54f9c6f49aa83c8ebae33844059": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "9cdcf372239b405d8ade6ba8f836ff9a": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "81d4531bc53646c0837de049e7e5fbf5": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "f7662da059d54f27ab620b474dff0650": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_8d6088d9f6634812a6bb9f7147125df0",
              "IPY_MODEL_dc44a1f2aa83478f8df181799ab60739",
              "IPY_MODEL_dea2ffabb6304887964eb1896fab372f"
            ],
            "layout": "IPY_MODEL_ab42d370004d4b38a17842445e21a8f7"
          }
        },
        "8d6088d9f6634812a6bb9f7147125df0": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_7ed618af34d14a81870b599adbf9e81e",
            "placeholder": "​",
            "style": "IPY_MODEL_c54ea00f47bc449fb68c1f445803b003",
            "value": "vocab.json: 100%"
          }
        },
        "dc44a1f2aa83478f8df181799ab60739": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_573ad370dbe04dcab40b4b7aad6cf305",
            "max": 1042301,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_91b0a6dcfa9f4cb688bef39992af6a2d",
            "value": 1042301
          }
        },
        "dea2ffabb6304887964eb1896fab372f": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_cdf49b7e09dc41ddbcb3e7166d0335cf",
            "placeholder": "​",
            "style": "IPY_MODEL_5173fd3a511f432ca6e70fb2678cc96a",
            "value": " 1.04M/1.04M [00:00&lt;00:00, 3.15MB/s]"
          }
        },
        "ab42d370004d4b38a17842445e21a8f7": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "7ed618af34d14a81870b599adbf9e81e": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "c54ea00f47bc449fb68c1f445803b003": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "573ad370dbe04dcab40b4b7aad6cf305": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "91b0a6dcfa9f4cb688bef39992af6a2d": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "cdf49b7e09dc41ddbcb3e7166d0335cf": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "5173fd3a511f432ca6e70fb2678cc96a": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "9089dad9898344d983f57adc8af08915": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_43716ace12be4aa9a8839c50cb44aece",
              "IPY_MODEL_e03368f9c21a4c67a0d750c6be80d944",
              "IPY_MODEL_59899125b3854783a4b2b12fab8ee696"
            ],
            "layout": "IPY_MODEL_8a8d4b5cd1bd4992bd3c445ba5125c0b"
          }
        },
        "43716ace12be4aa9a8839c50cb44aece": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_0375039625694d959478b310c332682d",
            "placeholder": "​",
            "style": "IPY_MODEL_ef6460b1ba2f4c85b18e19dc1c442696",
            "value": "merges.txt: 100%"
          }
        },
        "e03368f9c21a4c67a0d750c6be80d944": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_a6cb00ae08934b3fb6c9e2c7281abe73",
            "max": 456318,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_027ea360b5dd4ea7927718572e82f6ec",
            "value": 456318
          }
        },
        "59899125b3854783a4b2b12fab8ee696": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_f454d3cd4f654ae1892918083c53848c",
            "placeholder": "​",
            "style": "IPY_MODEL_59903b3255934075962d13952d12ac34",
            "value": " 456k/456k [00:00&lt;00:00, 2.78MB/s]"
          }
        },
        "8a8d4b5cd1bd4992bd3c445ba5125c0b": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "0375039625694d959478b310c332682d": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "ef6460b1ba2f4c85b18e19dc1c442696": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "a6cb00ae08934b3fb6c9e2c7281abe73": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "027ea360b5dd4ea7927718572e82f6ec": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "f454d3cd4f654ae1892918083c53848c": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "59903b3255934075962d13952d12ac34": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "8b40e529989747bf801eb200ce4446f4": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_58af7b6f315744f1b8f3564df9665e76",
              "IPY_MODEL_381aef476bb941fb9cc79c4812622475",
              "IPY_MODEL_fb961aaa596b44ea90f28b13f2008aaf"
            ],
            "layout": "IPY_MODEL_5a02aad4c21648a5838382f712933823"
          }
        },
        "58af7b6f315744f1b8f3564df9665e76": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_845df555f1464f5ba031590837dc555e",
            "placeholder": "​",
            "style": "IPY_MODEL_6a58411fc04d44cfac845eb131aa4d86",
            "value": "tokenizer.json: 100%"
          }
        },
        "381aef476bb941fb9cc79c4812622475": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_a7bda3966a4d4b7f9b3ffe0d7f04a9c5",
            "max": 1355256,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_70593da4e66d40a6abda62e1f573cddf",
            "value": 1355256
          }
        },
        "fb961aaa596b44ea90f28b13f2008aaf": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_784fdfcc81144072a0d252ff96bdae6f",
            "placeholder": "​",
            "style": "IPY_MODEL_8cf8a605f990424797b0866897a851fa",
            "value": " 1.36M/1.36M [00:00&lt;00:00, 3.30MB/s]"
          }
        },
        "5a02aad4c21648a5838382f712933823": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "845df555f1464f5ba031590837dc555e": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "6a58411fc04d44cfac845eb131aa4d86": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "a7bda3966a4d4b7f9b3ffe0d7f04a9c5": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "70593da4e66d40a6abda62e1f573cddf": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "784fdfcc81144072a0d252ff96bdae6f": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "8cf8a605f990424797b0866897a851fa": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "71e9270f47834a5bb1e039f1b5cd93d4": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_3ccf553f925b4687a123ca6899ff6ccd",
              "IPY_MODEL_7df4b432da8d405d9df10ff41f47eadd",
              "IPY_MODEL_96ed4c3587844972b2d33dc01efb094a"
            ],
            "layout": "IPY_MODEL_8ca702a4e3754a99b18978a6c781bb2b"
          }
        },
        "3ccf553f925b4687a123ca6899ff6ccd": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_a27d1cb808fb4bcc818123642b45d60c",
            "placeholder": "​",
            "style": "IPY_MODEL_1f25128e1fe9400aa84b1ea1f86d6d8f",
            "value": "Map: 100%"
          }
        },
        "7df4b432da8d405d9df10ff41f47eadd": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_5448634a10934fab99ac696ce7c52d21",
            "max": 1000,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_5a9d3cef4a404e448216de6532b5aab2",
            "value": 1000
          }
        },
        "96ed4c3587844972b2d33dc01efb094a": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_1b3de09bc8f3469d8aca84ab43e27d13",
            "placeholder": "​",
            "style": "IPY_MODEL_1d9801c7584a4ef382045cb87686a47f",
            "value": " 1000/1000 [00:00&lt;00:00, 1682.83 examples/s]"
          }
        },
        "8ca702a4e3754a99b18978a6c781bb2b": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "a27d1cb808fb4bcc818123642b45d60c": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "1f25128e1fe9400aa84b1ea1f86d6d8f": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "5448634a10934fab99ac696ce7c52d21": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "5a9d3cef4a404e448216de6532b5aab2": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "1b3de09bc8f3469d8aca84ab43e27d13": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "1d9801c7584a4ef382045cb87686a47f": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "bf916e995bd54c83b523f06908cf0642": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_ef8ad31cfe6b4cd2abd023413ef8e97f",
              "IPY_MODEL_69f081fb37ff4c6f853545b49b179106",
              "IPY_MODEL_1df417f3c1d94a2e8a9514f8b501e90a"
            ],
            "layout": "IPY_MODEL_27b6db172e3a45b1a6a2dc7e59a3bdc6"
          }
        },
        "ef8ad31cfe6b4cd2abd023413ef8e97f": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_f2378fc6533d4d37900b75c25b932452",
            "placeholder": "​",
            "style": "IPY_MODEL_2d2ea78ac132401dbd55c7dbe08fa266",
            "value": "Map: 100%"
          }
        },
        "69f081fb37ff4c6f853545b49b179106": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_cfa05e787eb04406b76936f1104be832",
            "max": 1000,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_008f7221bef74bdcadc5b06ecd0915f3",
            "value": 1000
          }
        },
        "1df417f3c1d94a2e8a9514f8b501e90a": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_46c99ea38fa6477699566077554d6347",
            "placeholder": "​",
            "style": "IPY_MODEL_bafd0dc85c6b429382bba264ca85e4e1",
            "value": " 1000/1000 [00:02&lt;00:00, 348.66 examples/s]"
          }
        },
        "27b6db172e3a45b1a6a2dc7e59a3bdc6": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "f2378fc6533d4d37900b75c25b932452": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "2d2ea78ac132401dbd55c7dbe08fa266": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "cfa05e787eb04406b76936f1104be832": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "008f7221bef74bdcadc5b06ecd0915f3": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "46c99ea38fa6477699566077554d6347": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "bafd0dc85c6b429382bba264ca85e4e1": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "55aaed8467354b0096a103ac4c6426a9": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HBoxModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_c62ec2f5655540498fb3fd761bf0fa35",
              "IPY_MODEL_a9d8dc65813f47ff9245c807381eea32",
              "IPY_MODEL_ca7f75bcb18445c5882fa7bd465a39f4"
            ],
            "layout": "IPY_MODEL_f89721670ba24234afa44af871ef8cb3"
          }
        },
        "c62ec2f5655540498fb3fd761bf0fa35": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_020fa2e561484519b7aee29a70ed9204",
            "placeholder": "​",
            "style": "IPY_MODEL_543025f54b0549eb9b1c1c165aa44127",
            "value": "Map: 100%"
          }
        },
        "a9d8dc65813f47ff9245c807381eea32": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "FloatProgressModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_4a2f531898c445798ff1e316c79368a2",
            "max": 1000,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_4b1d37ed14e549c9ae07e7e83dc61d36",
            "value": 1000
          }
        },
        "ca7f75bcb18445c5882fa7bd465a39f4": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "HTMLModel",
          "model_module_version": "1.5.0",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_e4a09d5fe9b0456f878669af845cf1ce",
            "placeholder": "​",
            "style": "IPY_MODEL_367a1c07c02f4b01a03af2a2bdf1cdbc",
            "value": " 1000/1000 [00:00&lt;00:00, 1039.29 examples/s]"
          }
        },
        "f89721670ba24234afa44af871ef8cb3": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "020fa2e561484519b7aee29a70ed9204": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "543025f54b0549eb9b1c1c165aa44127": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "4a2f531898c445798ff1e316c79368a2": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "4b1d37ed14e549c9ae07e7e83dc61d36": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "ProgressStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "e4a09d5fe9b0456f878669af845cf1ce": {
          "model_module": "@jupyter-widgets/base",
          "model_name": "LayoutModel",
          "model_module_version": "1.2.0",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "367a1c07c02f4b01a03af2a2bdf1cdbc": {
          "model_module": "@jupyter-widgets/controls",
          "model_name": "DescriptionStyleModel",
          "model_module_version": "1.5.0",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        }
      }
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
