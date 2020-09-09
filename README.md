# Deep Learning for Computer Vision's Practice

## Description:
Code examples from Adrian Rosebrock's *Deep Learning for Computer Vision with Python*.

<details><summary>Link to download</summary>

[Click here](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)
</details>

## Copyright:
```
@book{rosebrock_dl4cv,
  author={Rosebrock, Adrian},
  title={Deep Learning for Computer Vision with Python},
  year={2019},
  edition={3.0.0},
  publisher={PyImageSearch.com}
}
```

##Considerations:
It's important to say that all codes in this repo are inspired/a copy from the examples in the book and that I shouldn't take credit for any of it.
> **NOTE:** The main goal of this repository is to be a guide and exemplify my knoledge throughout the world of Deep Learning and Computer Vision, working as baseline.



## Quickstart;
- Run the `docker_run.sh` file using the path to the python file as parameter as bellow:
    ```$ docker_run.sh PATH/TO/MY/PYTHON/file.py```
> **NOTE:** If this is the first time you run the `docker_run.sh` file and you haven't already pulled the `diogo364/dl4cv:1.0` image from my DockerHub it will automatically pull the image and might take a while.
- You will be asked to enter all CLI parameters for the python script you are running as shown bellow:
    ```
    ...
    > [INSTRUCTION] Enter parameters CLI parameters, else press enter!
    > [TIP] If you don't know about CLI parameters type -h or --help
    $ --d CLI-PARAMETER1 -e CLI-PARAMETER2
    ```
- After the execution all images outputs will be saved inside an outputs dir within your workdir