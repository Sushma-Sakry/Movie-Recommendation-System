{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOpWlXFeYqD096e7jIN5ywJ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Sushma-Sakry/Movie-Recommendation-System/blob/main/mean_var_std.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "def calculate(list):\n",
        "\n",
        "    if len(list) != 9:\n",
        "        raise ValueError(\"List must contain nine numbers.\")\n",
        "\n",
        "    matrix = np.array(list).reshape(3, 3)\n",
        "\n",
        "    calculations = {\n",
        "        'mean': [\n",
        "            matrix.mean(axis=0).tolist(),\n",
        "            matrix.mean(axis=1).tolist(),\n",
        "            matrix.mean().tolist()\n",
        "        ],\n",
        "\n",
        "        'variance': [\n",
        "            matrix.var(axis=0).tolist(),\n",
        "            matrix.var(axis=1).tolist(),\n",
        "            matrix.var().tolist()\n",
        "        ],\n",
        "\n",
        "        'standard deviation': [\n",
        "            matrix.std(axis=0).tolist(),\n",
        "            matrix.std(axis=1).tolist(),\n",
        "            matrix.std().tolist()\n",
        "        ],\n",
        "\n",
        "        'max': [\n",
        "            matrix.max(axis=0).tolist(),\n",
        "            matrix.max(axis=1).tolist(),\n",
        "            matrix.max().tolist()\n",
        "        ],\n",
        "\n",
        "        'min': [\n",
        "            matrix.min(axis=0).tolist(),\n",
        "            matrix.min(axis=1).tolist(),\n",
        "            matrix.min().tolist()\n",
        "        ],\n",
        "\n",
        "        'sum': [\n",
        "            matrix.sum(axis=0).tolist(),\n",
        "            matrix.sum(axis=1).tolist(),\n",
        "            matrix.sum().tolist()\n",
        "        ]\n",
        "    }\n",
        "\n",
        "    return calculations"
      ],
      "metadata": {
        "id": "HObMUGwANXob"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(calculate([0,1,2,3,4,5,6,7,8]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FCy2vdZrNce_",
        "outputId": "67828c10-639f-448a-d9a1-3e1324b5dafa"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'mean': [[3.0, 4.0, 5.0], [1.0, 4.0, 7.0], 4.0], 'variance': [[6.0, 6.0, 6.0], [0.6666666666666666, 0.6666666666666666, 0.6666666666666666], 6.666666666666667], 'standard deviation': [[2.449489742783178, 2.449489742783178, 2.449489742783178], [0.816496580927726, 0.816496580927726, 0.816496580927726], 2.581988897471611], 'max': [[6, 7, 8], [2, 5, 8], 8], 'min': [[0, 1, 2], [0, 3, 6], 0], 'sum': [[9, 12, 15], [3, 12, 21], 36]}\n"
          ]
        }
      ]
    }
  ]
}