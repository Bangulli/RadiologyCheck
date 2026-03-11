# PARSOR concept demo
This repository contains the concept demonstrator for PARSOR funded by Innosuisse no. 129.858 INNO-LS

## Approach
PARSOR aims at automating the full radiology reporting and diagnosis process and limiting it to only one human interaction where the radiologist reviews the AI generated findings and report. Due to data limitations no such model could be trained however the workload of reporting remains tedious and time consuming for the professionals.
With this PARSOR demo it is possible to automate the reporting in one prompt by automating translation and formatting in code. 
The user is prompted to provide the subspecialty, patient history and their findings in german or english. The results are then processed and translated by Helsinki-NLP's opus, then parsed into a unified prompt and fed to Google's MedGemma for report generation.

<img src="better parsor demo concept.png" width="1000px" align="center" />  


## Requirements
This repository makes use of open source models from the Huggingface model zoo.
Make sure to request access to the following models:
- google/medgemma-1.5-4b-it
- Helsinki-NLP/opus-mt_tiny_deu-eng

## Building
To build this project locally as a Docker image create a `token.txt` file in the work directory containing your huggingface access token with permissions to access the models stated above.
The huggingface token is passed to the docker builder as a secret at build time. 
The baseprompt is copied into the container at build time, make sure to update the filename of your prompt in both the config ("baseprompt") and the dockerfile (line 56).
To build run:
```bash
docker build --secret id=hf_token,src=token.txt -t parsor:latest .
```

## Running
Model outputs are stored as markdown (.md) files. To run mount an output directory and specify an output filepath with the `-dir` flag:
```bash
docker run --rm -it --gpus all -v ./outputs:/outputs parsor:latest -dir /outputs/run
```
The container is designed to interact with the user and requests the radiologist subspecialty, patient information, radiologist findings as well as any supplementary files at runtime. Note that only text files are supported, and that textfiles have to be stored in a directory and mounted to the container as well using `-v YOUR_INPUT_DIR:/inputs` then the filepaths are available to the pipeline at the path `/inputs/YOUR_FILE_NAME`