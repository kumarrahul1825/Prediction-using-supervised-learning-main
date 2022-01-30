{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Aditya Srivastava\n",
    "## Task-1 - Prediction using supervised learning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Importing important libraries---\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data is successfully imported\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1.5</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>9.2</td>\n",
       "      <td>88</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>5.5</td>\n",
       "      <td>60</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8.3</td>\n",
       "      <td>81</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2.7</td>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>7.7</td>\n",
       "      <td>85</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>5.9</td>\n",
       "      <td>62</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>4.5</td>\n",
       "      <td>41</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>3.3</td>\n",
       "      <td>42</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>1.1</td>\n",
       "      <td>17</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>8.9</td>\n",
       "      <td>95</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>2.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>1.9</td>\n",
       "      <td>24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>6.1</td>\n",
       "      <td>67</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>7.4</td>\n",
       "      <td>69</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>2.7</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>4.8</td>\n",
       "      <td>54</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>3.8</td>\n",
       "      <td>35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>6.9</td>\n",
       "      <td>76</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>7.8</td>\n",
       "      <td>86</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Hours  Scores\n",
       "0     2.5      21\n",
       "1     5.1      47\n",
       "2     3.2      27\n",
       "3     8.5      75\n",
       "4     3.5      30\n",
       "5     1.5      20\n",
       "6     9.2      88\n",
       "7     5.5      60\n",
       "8     8.3      81\n",
       "9     2.7      25\n",
       "10    7.7      85\n",
       "11    5.9      62\n",
       "12    4.5      41\n",
       "13    3.3      42\n",
       "14    1.1      17\n",
       "15    8.9      95\n",
       "16    2.5      30\n",
       "17    1.9      24\n",
       "18    6.1      67\n",
       "19    7.4      69\n",
       "20    2.7      30\n",
       "21    4.8      54\n",
       "22    3.8      35\n",
       "23    6.9      76\n",
       "24    7.8      86"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Importing dataset\n",
    "path =  \"http://bit.ly/w-data\"\n",
    "Data = pd.read_csv(path)\n",
    "print(\"Data is successfully imported\")\n",
    "Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.5</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.1</td>\n",
       "      <td>47</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.2</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.5</td>\n",
       "      <td>75</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.5</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours  Scores\n",
       "0    2.5      21\n",
       "1    5.1      47\n",
       "2    3.2      27\n",
       "3    8.5      75\n",
       "4    3.5      30"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Now print the first 5 records...\n",
    "\n",
    "Data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>2.7</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>4.8</td>\n",
       "      <td>54</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>3.8</td>\n",
       "      <td>35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>6.9</td>\n",
       "      <td>76</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>7.8</td>\n",
       "      <td>86</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Hours  Scores\n",
       "20    2.7      30\n",
       "21    4.8      54\n",
       "22    3.8      35\n",
       "23    6.9      76\n",
       "24    7.8      86"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "##Now print the last 5 records...\n",
    "Data.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours</th>\n",
       "      <th>Scores</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>25.000000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.012000</td>\n",
       "      <td>51.480000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.525094</td>\n",
       "      <td>25.286887</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.100000</td>\n",
       "      <td>17.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.700000</td>\n",
       "      <td>30.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>4.800000</td>\n",
       "      <td>47.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.400000</td>\n",
       "      <td>75.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>9.200000</td>\n",
       "      <td>95.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Hours     Scores\n",
       "count  25.000000  25.000000\n",
       "mean    5.012000  51.480000\n",
       "std     2.525094  25.286887\n",
       "min     1.100000  17.000000\n",
       "25%     2.700000  30.000000\n",
       "50%     4.800000  47.000000\n",
       "75%     7.400000  75.000000\n",
       "max     9.200000  95.000000"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#here we use describe() method so that we can able to see percentiles,mean,std,max,count of the given dataset.\n",
    "Data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25 entries, 0 to 24\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype  \n",
      "---  ------  --------------  -----  \n",
      " 0   Hours   25 non-null     float64\n",
      " 1   Scores  25 non-null     int64  \n",
      "dtypes: float64(1), int64(1)\n",
      "memory usage: 464.0 bytes\n"
     ]
    }
   ],
   "source": [
    "#Let's print the full summary of the dataframe .\n",
    "Data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "##importing libraries for plotting Graphs\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualizing Data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAe0AAAFlCAYAAADGV7BOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3dfXxTZZ738W+SSkpCC8VpwdsCQ7VWwdfuKgwgAoqDFqkoIs8OOOITLO5QV4VCoVUBseowg6iAKMsslSflcYTVZSoOIL4q660sr4rKk1uKlDJShzZtQ5vk/qM3EbalFMppzkk+7784OUnO72pKv7mu65zr2AKBQEAAAMD07KEuAAAANA6hDQCARRDaAABYBKENAIBFENoAAFgEoQ0AgEUQ2ghLKSkpOnny5DmPrVu3Tk888USz11JeXq4ZM2Zo8ODBuvfeezVkyBC99957wf3vvfee3n333Yt+33vuuUf5+fk6fvy4Ro0adcmvv9w++eQTjRw5Uvfee6/S0tI0efJkFRcXX/bjNEZpaalmzZql1NRU3XPPPRowYICef/55lZeXX9bj1Pf7BhiB0AYM9vvf/14ul0ubNm3Spk2btHjxYr3xxhvauXOnJOmLL75QVVXVJb9/u3bttGrVqstVbpMcP35cU6dO1bx587Rp0yZt3rxZ119/vdLT05u9lvLyco0aNUpxcXH64IMP9MEHH2jLli2y2+165plnmr0e4HKICnUBQCiUlZXp+eef1zfffCObzaa+ffvqX//1XxUVFaWUlBR99tlnatu2rSQFt/fv3685c+bI5XLJ4/FoxYoVyszM1P/8z//Ibrera9eueuGFF2S3n/td+MSJE7ryyitVXV2tFi1aqF27dlqwYIHatGmjrVu36uOPP9ann36q6OhonTx5UqWlpcrKypIkLViwILh94MABTZ8+XZWVlUpKSlJFRYUkqaioSIMHD9aXX34pSVq4cKH+8z//U36/X1dffbWys7PVrl27877+bIcPH9aoUaO0Y8cOtWjRQj6fT7fffruWLVumgwcPauHChbLZbHI4HJoyZYp+9atfnfP60tJSVVdXn/PeDz30kK6//vrg9uLFi7V+/XpFRUWpU6dOeumllxQTE6M33nhDmzdvlsPhUOfOnTVz5kzFx8dr7Nixat26tQ4dOqTRo0dryJAhmjNnjr777jtVV1frlltu0ZQpUxQVde6fszVr1uiXv/ylnnzyyeBjLVq00JQpU/TOO+/I7/dr9+7d53yma9eu1csvv6w9e/bI4/EoEAho9uzZ6tatmzIyMuR0OvXNN9/oxx9/1K233qoZM2boiiuuCH5We/bs0U8//aRHHnlEDz744MX9UgKNQGgjbD300EPnBOjf//53paSkSJJmz56tNm3a6M9//rOqq6s1ceJELV26VI8//niD77l//3795S9/0dVXX60NGzbI4/Fo48aN8vl8ys7O1pEjR9SpU6dzXvPkk09q8uTJ6tWrl2666SbdfPPNGjRokDp06KAOHTooLy9PycnJevDBB7VgwYLzHvuZZ57Rgw8+qOHDh+uLL76oNxQ2bNig7777Tu+9956ioqK0evVqzZgxQ0uWLGnU6zt37qzk5GR9/PHHGjhwoHbu3KnExERdc801euKJJ/Tqq6/qn/7pn7Rz507l5+fXCe3rr79eI0aM0P3336+OHTvq5ptv1i233KLU1FRJUl5entatW6c1a9aodevWmjt3rnJzc5WQkKAdO3bo/fffl8vl0oIFC5SRkaF33nlHkhQbG6stW7ZIkqZNm6auXbvqpZdeks/nU0ZGhv7t3/5Njz322Dm1/Nd//Zf69OlTp41Op1P//M//XO9n+uWXX6qkpESrV6+W3W7XW2+9pSVLlqhbt26SpP/+7/9Wbm6urrjiCo0fP16rV6/Wb37zG0lShw4dlJ2dra+//lojR47UiBEjgoEOXC6ENsLWn/70p2BvWaqd0/7oo48kSdu3b9fKlStls9nUokULjRo1Sn/6058uGNpXXXWVrr76aklSt27d9Ic//EFjx45V79699dBDD9UJbKk2yD788EMVFBRo9+7d+vTTT7Vo0SLNnz9fd9xxR6PaUlpaqm+//VZDhgwJHjs5ObnO87Zt26a9e/fqgQcekCT5/X5VVlY2+vWSNGzYMK1fv14DBw7UunXrNGLECElSWlqannzySd1222269dZb64TkGRkZGXriiSf0+eefa/fu3Xr55Ze1fPlyvfvuu/rss880cOBAtW7dWlJtAEvS5MmTNXToULlcLknSuHHjtGjRIp0+fVqS1L179+D7f/LJJ9q7d6/ef/99STrv1EIgEJDNZgtub9q0Kfgl4OTJk1qyZImkcz/Tm266Sa1bt9aqVat05MgR5efny+12B9/j/vvvD27fd999ysvLC4b2PffcI0m64YYbdPr0aZWXlysuLq7e2oBLxZw2IpLf7z/nD7rf71dNTU2d550JjTPOhIpU27PaunWrHn/8cZWXl+vhhx/Wxx9/fM7za2pqlJWVpb///e+68cYb9fDDD+vtt9/WxIkTtXr16jrHs9lsOvt2ANXV1efsP3vf/x4OPtOORx99VBs3btTGjRu1du1arVy5stGvl6S7775be/bs0cGDB7V7924NHDhQkvTUU09pxYoVuvHGG7Vu3bp6e+p5eXlau3at4uLilJqaqhkzZmjLli06cOCAvv76azkcjnN+7qdOnVJRUdEFP4+zf+5+v1/z588PtvG9994LTiec7aabbtLnn38e3L733nuDr7niiiuCP9uz3/uTTz4Jnqz461//WqNHjz7nPR0Oxzk/y7NHcs78PM+0g9s6wAiENiJSnz59lJubq0AgoNOnT2vNmjXq3bu3JKlt27bau3evJOmDDz4473usWLFC06ZNU58+ffTss8+qT58++vrrr895TlRUlA4fPqw333wzGBI1NTU6ePCgunTpIqk2CM4EVFxcnAoKChQIBFReXq5t27YFH+/atWvwrPOCggJ999139bbr/fffD54dPX/+fE2ZMqXRr5dqh4/T0tKUkZGhu+66Sy1btlRNTY3uuOMOVVZWavTo0crOzta3335b50uN2+3WvHnzdODAgeBjR44ckcPhUMeOHdW7d29t3bo1WN+CBQu0bNky9e3bV2vXrg3OhS9fvly/+tWv1KJFi3rbuGzZsuBnN3HiROXm5tZ53pgxY3TgwAG9/fbbwTr9fr927typn3766ZwAPuPTTz9V//79NWbMGN144436y1/+Ip/PF9z/H//xHzp9+rS8Xq/Wr1+v/v371/szBIzC8Dgi0owZMzR79mwNHjxY1dXV6tu3ryZMmBDc98ILLyg2Nla9e/dWfHx8ve8xZMgQff755xo0aJBatmypq666SmPHjq3zvPnz5+uVV15RamqqWrZsKb/frzvvvFOTJk2SJPXr108vvfSSpNqg2bFjh+666y61a9dOPXr0CPbY5s2bp2nTpmnVqlXq2LGjkpKS6hxr+PDhOn78uEaMGCGbzaarrroq+N6Nef3Z75Obm6vnnntOUu2Xj+nTp+uZZ55RVFSUbDabXnzxxTqh2qtX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ZWVl6Ze//GWoywoKy9BesmSJNm3apJYtW4a6lEu2adMmtWnTRq+88opKS0t1//33Wza0t23bJklatWqV8vPzNXfuXC1cuDDEVV286upqZWVlKTo6OtSlNInX65Uky36hPSM/P19ffvmlVq5cqcrKSi1dujTUJV2SoUOHaujQoZKk559/Xg888IDlAluS/vrXv6qmpkarVq3Sp59+qj/+8Y9asGBBqMu6aGvWrJHL5dKaNWt06NAhzZo1S++8806oywoKy+Hxjh07WvKX5WwDBw7U5MmTg9sOhyOE1TTNgAEDNGvWLEnSDz/8oF/84hchrujS5OTkaNSoUUpISAh1KU3yzTffqLKyUuPHj9e4ceP01VdfhbqkS7Jz505dd911mjRpkiZMmKDbb7891CU1yd69e3XgwAGNHDky1KVcks6dO8vn88nv96u8vFxRUdbsEx44cED9+vWTJCUlJengwYMhruhc1vypXkBqaqqKiopCXUaTuN1uSVJ5ebl+97vfKT09PcQVNU1UVJSmTp2qrVu36rXXXgt1ORdt3bp1atu2rfr27au33nor1OU0SXR0tB555BENHz5c33//vR577DF9+OGHlvsjW1paqh9++EGLFi1SUVGRJk6cqA8//FA2my3UpV2SxYsXa9KkSaEu45K5XC4dPXpUd999t0pLS7Vo0aJQl3RJbrjhBm3btk0DBgzQnj17dPz4cfl8PtN0nMKypx0ujh07pnHjxum+++7T4MGDQ11Ok+Xk5Oijjz7SzJkzVVFREepyLsratWu1a9cujR07Vvv27dPUqVN14sSJUJd1STp37qx7771XNptNnTt3Vps2bSzZljZt2qhPnz5q0aKFkpKS5HQ6dfLkyVCXdUlOnTqlQ4cOqVevXqEu5ZItW7ZMffr00UcffaSNGzcqIyMjOBVjJQ888IBatWqlcePGadu2beratatpAlsitE3rb3/7m8aPH69nn31Ww4YNC3U5TbJhwwYtXrxYktSyZUvZbDZT/SdojHfffVe5ublavny5brjhBuXk5Cg+Pj7UZV2S999/Xy+99JIk6fjx4yovL7dkW7p166YdO3YoEAjo+PHjqqysVJs2bUJd1iXZvXu3evfuHeoymiQ2NjZ4cmPr1q1VU1Mjn88X4qou3t69e9WtWzctX75cAwYMUIcOHUJd0jmsNR4WQRYtWqRTp07pzTff1Jtvvimp9gQ7K54Eddddd2natGl68MEHVVNTo+nTp8vpdIa6rIg1bNgwTZs2TaNHj5bNZtOLL75ouaFxSerfv792796tYcOGKRAIKCsry3JfBs84fPiwEhMTQ11Gk/z2t7/V9OnTNWbMGFVXV+upp56Sy+UKdVkXrVOnTpo/f76WLl2qmJgYzZkzJ9QlnYO7fAEAYBEMjwMAYBGENgAAFkFoAwBgEYQ2AAAWQWgDAGARhDYQ5oqKinTHHXfUeTwlJSUE1QBoCkIbAACLILSBCOb3+zV79mylpaXpnnvuCa6rnp+fr7Fjxwafl5GRoXXr1qmoqEgDBw7U6NGj9fDDD+ubb77RiBEjNHToUI0ePVrff/99iFoCRAbrLYME4KKVlJTovvvuq/P4ypUrdezYMW3atEmnT5/W2LFjdd111zV4W9vDhw/r7bffVmJioqZNm6aHH35Yd999t9avX6+vvvrKVPceBsINoQ1EgISEBG3cuPGcx1JSUpSfn6/7779fDodDLVu21ODBg/XZZ5/VOwd+xpVXXhlccvO2227TCy+8oB07duiOO+5Q//79DW0HEOkYHgcimN/vP2c7EAjI5/PJZrPp7BWOq6urg/8+e/37gQMHav369fqHf/gHLVu2TNnZ2cYXDUQwQhuIYL169dKGDRvk8/lUWVmpP//5z+rZs6fi4uJ05MgReb1e/fTTT/riiy/qfX16err27t2rUaNGafLkyfr666+buQVAZGF4HIhgI0eO1Pfff6/77rtP1dXVGjx4sO68805JtUPfaWlpuvrqq9WtW7d6Xz9hwgRlZmbqjTfe0BVXXKHnnnuuGasHIg93+QIAwCIYHgcAwCIIbQAALILQBgDAIghtAAAsgtAGAMAiCG0AACyC0AYAwCIIbQAALOL/AUYPr1XD1YQHAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualise\n",
    "plt.style.use('seaborn')\n",
    "plt.scatter(Data.Hours,Data.Scores,color='orange')\n",
    "plt.title(\"Hours Studied vs Score Graph\")\n",
    "plt.xlabel(\"Hours\")\n",
    "plt.ylabel(\"Score\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## This \"SCATTER PLOT\" indicates positive linear relationship as much as hours You study is a chance of high scoring"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2.5],\n",
       "       [5.1],\n",
       "       [3.2],\n",
       "       [8.5],\n",
       "       [3.5],\n",
       "       [1.5],\n",
       "       [9.2],\n",
       "       [5.5],\n",
       "       [8.3],\n",
       "       [2.7],\n",
       "       [7.7],\n",
       "       [5.9],\n",
       "       [4.5],\n",
       "       [3.3],\n",
       "       [1.1],\n",
       "       [8.9],\n",
       "       [2.5],\n",
       "       [1.9],\n",
       "       [6.1],\n",
       "       [7.4],\n",
       "       [2.7],\n",
       "       [4.8],\n",
       "       [3.8],\n",
       "       [6.9],\n",
       "       [7.8]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = Data.iloc[:,:-1].values\n",
    "Y = Data.iloc[:,1].values\n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30,\n",
       "       24, 67, 69, 30, 54, 35, 76, 86], dtype=int64)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Preparing Data and splitting into train and test sets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0,test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X train.shape = (20, 1)\n",
      "Y train.shape = (20,)\n",
      "X test.shape  = (5, 1)\n",
      "Y test.shape  = (5,)\n"
     ]
    }
   ],
   "source": [
    "## We have Splitted Our Data Using 80:20 RULe(PARETO)\n",
    "print(\"X train.shape =\", X_train.shape)\n",
    "print(\"Y train.shape =\", Y_train.shape)\n",
    "print(\"X test.shape  =\", X_test.shape)\n",
    "print(\"Y test.shape  =\", Y_test.shape)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training the Model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "linreg=LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training our algorithm is finished\n"
     ]
    }
   ],
   "source": [
    "##Fitting Training Data\n",
    "linreg.fit(X_train,Y_train)\n",
    "print(\"Training our algorithm is finished\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "B0 = 2.018160041434669 \n",
      "B1 = [9.91065648]\n"
     ]
    }
   ],
   "source": [
    "print(\"B0 =\",linreg.intercept_,\"\\nB1 =\",linreg.coef_)## β0 is Intercept & Slope of the line is β1.,\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "##plotting the REGRESSION LINE---\n",
    "Y0 = linreg.intercept_ + linreg.coef_*X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAe0AAAFlCAYAAADGV7BOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3de3gTZd4+8DuHNj2kadM2xQPIOQieqCAgFgqllbIuiwJSqSu6vK+uiK74U5QiB114RYTVFV226uqqnJWDsCd3gSIoYMWlKgJaK4K0QEnbtOmBpmkyvz+QSEjapm0mM5Pcn+t6r8s8k8x8p33Zu995JvOoBEEQQERERLKnlroAIiIi8g9Dm4iISCEY2kRERArB0CYiIlIIhjYREZFCMLSJiIgUgqFNYa+0tBT9+/fHhAkT3P/3q1/9Chs3buz0vn/7299i8+bNAIAJEybAZrO1+N7a2lpMmzbN/bqt9wfbvHnz8PXXX7frM+Xl5bjrrrvafN/999+PkpKSjpbWqot/By259GdPJFdaqQsgkoOoqChs3brV/bq8vBy//OUvce211+Lqq68OyDEu3r8vNTU1OHTokN/vD7Z9+/YhJyenXZ/p0qUL1q9f3+b73njjjY6WFRCX/uyJ5IqhTeRDly5d0L17dxw/fhxHjhzBxo0bce7cOej1eqxatQrvv/8+1q1bB5fLhYSEBMyfPx+9e/dGeXk55syZg7Nnz+KKK65AZWWle5/9+vXD/v37kZiYiNdeew1btmyBVqtF9+7d8fzzzyMvLw+NjY2YMGECNm/ejAEDBrjf/6c//Qn/+Mc/oNFo0LNnT8yfPx8mkwn33HMPBg4ciIMHD+L06dO4+eabsWjRIqjVnhfRzpw5g2eeeQZlZWUQBAG33347/vd//xelpaW47777kJ6eji+//BI2mw2zZ89GVlaWx+dfeuklnD17Fk888QReeOEFLF++HPHx8Th27BimTp2K6667DsuWLUNTUxMsFguGDx+O5557DqWlpRg/fjyKiorwyiuvoKysDBaLBWVlZejSpQuWLVuGlJQUZGRk4OWXX0ZDQwNeeukldOvWDd999x2am5vx7LPPYtCgQaiqqkJeXh5+/PFHJCQkwGQyoW/fvnjkkUc8am3td7Bx40Zs2LABDocDNTU1uP/++5Gbm+v1s9+yZYvP9xFJTiAKcydPnhQGDhzoMXbw4EHhpptuEk6dOiVs2rRJuOmmm4Ta2lpBEAShsLBQyM3NFRoaGgRBEISPP/5YyM7OFgRBEB566CHhpZdeEgRBEI4fPy4MHDhQ2LRpkyAIgmA2m4XKykphx44dwq233ipUV1cLgiAIzz33nLBy5UqvOi68f+PGjUJOTo5QX18vCIIgrFixQpg+fbogCILw61//Wvjd734nOJ1Ooba2VkhLSxP279/vdY5333238NZbbwmCIAg2m00YP3688Pe//104efKkYDabhYKCAkEQBOHDDz8URo0a5fPnNHr0aOGrr75yHzcvL8+97bHHHhM+/fRTQRAEoa6uThg6dKhw6NAhj3NasWKFMGbMGPfP8be//a3w8ssve+z7008/Ffr37y8cOXJEEARBePPNN4W7777bfYwXXnhBEARBKC8vF2655RZhxYoVXnW29Duoq6sTpkyZIlRVVQmCIAhFRUXu2i6us7X3EUmNnTYR4O6yAMDpdMJoNGLZsmW4/PLLAZzvkvV6PQDgo48+wokTJzzmam02G6qrq7Fv3z489dRTAIDu3btj6NChXsfav38/srOzER8fDwDIy8sDcH5u3Zc9e/Zg4sSJiImJAQBMmzYN+fn5aGpqAgCMHj0aarUaer0e3bt3R01NjcfnGxoacPDgQbz11lsAgLi4OEycOBF79uzBDTfcgIiICKSnpwMABgwYgOrqar9+ZoMHD3b/9/PPP489e/YgPz8fx44dg91uR0NDAxISEjw+M2TIEPfPccCAAV61AsAVV1yB/v37u9+zZcsWAMDu3bvd/52SkoLs7GyfdbX0O4iNjUV+fj52796N48eP45tvvkFDQ4PX5/19H5EUGNpE8J7TvtSFwAQAl8uFCRMmYPbs2e7XZ8+eRXx8PFQqFYSLHuev1Xr/E9NoNFCpVO7XNput1RvOXC6Xx/tdLheam5s9ar/g0uNfeL+vsQv7iIiIcF9Ov/g4bbn4Z/LrX/8a/fr1w4gRIzBu3Dh8+eWXXsf0p9bW3qPVaj3ef+kUQEv7vfA7OHPmDHJycjBlyhQMGjQI2dnZ2LVrl9fn/X0fkRR49zhRO6WlpeEf//gHzp49CwBYt24d7r33XgDAiBEjsGHDBgDAqVOnUFhY6PX54cOHY/v27airqwMAvPLKK3j77beh1WrhdDq9gmzEiBHYtGmTu9tbtWoVbrrpJkRGRvpVr16vxw033IA1a9YAOH+n9AcffIDhw4e367w1Go3HHwsX2Gw2HDp0CE888QRuvfVWnDlzBj/++CNcLle79t+W9PR09x39VqsVO3bs8PlHRku/g6+//hqJiYl46KGHkJaW5g5ip9Pp8bNv7X1EUmOnTdROaWlpuP/++zF9+nSoVCro9Xq8+uqrUKlUWLhwIfLy8jBu3DhcdtllPu88T09PR0lJCaZOnQoA6NOnDxYtWoTo6Ghcf/31uO2229wBCwCTJ0/G6dOnceedd8LlcqF79+5Yvnx5u2pevnw5fv/732Pz5s1oamrC+PHjMXHiRJSVlfm9j6ysLMyePRvPPPOMx7jBYMADDzyAO+64AzExMejSpQtuvPFGnDhxAt26dWtXna3Jy8vDvHnzMH78eCQkJOCKK67w6MovaOl3cMstt2Djxo3Izs6GSqXCkCFDkJiYiBMnTqB79+7un/1f//pXdOnSxef7evXqFbDzIeoIleDr+hQRkcysWbMGAwYMQGpqKpqampCbm4tHHnnEPR9PFA7YaRORIly4IuFyueBwOJCdnc3AprDDTpuIiEgheCMaERGRQjC0iYiIFIKhTUREpBCyvhHNYqnt1OeNxhhYrcp/khHPQ154HvLC85CXUDkPQLpzMZniWtwW0p22VquRuoSA4HnIC89DXnge8hIq5wHI81xCOrSJiIhCCUObiIhIIRjaRERECsHQJiIiUgiGNhERkUIwtImIiBSCoU1ERKQQsn64ilwdPPg5FizIQ48ePaFSqWC323HrrdmYPPmudu3nz39+Bd2790DfvmZ88ske/OY39/t83/bt29G1a2+oVCr89a9/wRNPzAnEaRAREQC7w4maOjvi9TroIuT33eyLMbQ7aNCgwXj22SUA8NPavpMwduxtiItr+Uk2Lenbtx/69u3X4vZ3330Xjz76JLp378HAJiIKEKfLhQ0FJSgqtqDKZkeiQYdUswk5GX2gUcvzQrSiQzu2eB505R+0/AaNConO9q08au9yO+rNi9v1mYaGBqjVasya9RAuv/wK1NbWYtmyP+IPf3gepaUn4XK5cP/9M3DjjYPx0Uc78c47byIhwQiHw4Hu3Xvg4MHPsXXrJjz77BL8/e8fYMuWTXC5nEhLS0f//tfg6NGjWLx4AebPX4TFixfi9dffxoEDn+L11/8MnU4HgyEeeXkL8N1332LNmncREaHF6dOnkJGRhXvv/Z92nQsRUbjYUFCCHZ+Xul9X2uzu17mZZqnKapWiQ1tK//3v53j44QegVquh1Wrx2GOzsWbNu8jKykZ6+mhs2bIR8fEJyMtbgJqaasyc+QBWr34PK1euwBtvvAODIR6zZz/qsU+rtQqrV7+Dd95Zh4iISLz66ksYOPBG9O/fH48++iQiIiIAAIIg4IUXnsPKlX+ByZSC995bh3feeRPDh6ehvPw03n57HRwOB26/PZuhTUTkg93hRFGxxee2ouIKTErvHeSK/KPo0K43L261KzaZ4lDVyUVHWnLx5fEL1qx5F1dd1R0A8P33JfjqqyIcOfI1AMDpbEZVVSViY2MRH58AALj22us9Pl9WVoaePXtDp4sCAPzud4/7PHZ1dTViYmJhMqUAAAYOTMVrr63E8OFp6NWrD7RaLbRarXs/RETkqabOjiqb3ec2a20jaurs6Brkmvwhz4v2Cqb+aR6ke/ceyMwci1dffR1/+MMKjB6dibg4A+rq6mG1WgEA33xzxOOzV17ZFT/+eBxNTU0AgHnznoTFchYqlQoul8v9voSEBDQ01KOiogIA8MUXB9Gt21UAAJVK9FMkIlK8eL0OiQadz23GuCjE631vk5qiO205mzBhIpYuXYyHH34A9fV1uOOOOxEREYG5cxfg8ccfRlxcPLRazx+/0WjE3Xffi4cffgAqlQq33DICJlMKUlNTsXjxQjz55NMAAJVKhSeffBpPPz0barUKcXEGzJ37DI4dK5HiVImIFEcXoUGq2eQxp31BqjlZtneRqwRBaN+dWkHU2fW0Taa4Tu9DDnge8sLzkBeeh7wo6Tx+vnu8AtbaRhjjopBqTnbfPS7VubS2njY7bSIiCksatRq5mWZMSu+tmO9pc06biIjCmi5CgxRjTLsDW91wDNE/vASVo0akyryx0yYiImqnuK8fQNTp9QAAZ0xvNHX5VVCOy9AmIiLyk6buCBL3D3O/dmkT0JRyW9COz9AmIiJqiyDA8MUU6Cr+7R6quWE9mlJ+EdQyGNpERESt0NZ8DuNnGe7XzdG9YB1+AFBHBL+WoB+RiIhICQQXEj7LRETNZ+6h6hu3wZE0SrKSGNpERESXiKj6GNh+Gy700g7DjageUgCopP3SFUObiIjoAlczjPuHQtvwnXvIetMONCcMkbConzG0iYiIAERaPkT8F1N+Hrh8LCzXvCerRR0Y2kREFN5cdiTt6Q+1o8I9VDVsHxJ73QzI7JGsDG0iIgpbhi+mQmf5h/t1Y5dJqL3+rxJW1DqGNhERhR1VUwWSd/fyGKsa/l84Y/tKVJF/GNpERBRWEgpHIcJ20GPMkmWTqJr2YWgTEVFYUJ/7EUmfXOsxZr3pP2hOGNbCJ+SHoU1ERCEvcXdfaJrKPcaU0l1fjKFNREQhS1N3FIn7h3qMVQ3bD2fcNQHZv93hDOpa3AxtIiIKSabtBo/XrogkVI76ISD7drpc2FBQgqJiC6psdiQadEg1m5CT0QcatXhPTWNoExFRSNFWF8J4IMtjrDLtK7iiewTsGBsKSrDj89Kf92+zu1/nZpoDdpxLSfsQVSIiogAybTd4BLYj7gZYsmwBDWy7w4miYovPbUXFFbA7nAE71qXYaRMRkeJFVGxHQtEkj7GKkSUQdCkBP1ZNnR1VNrvPbdbaRtTU2ZFijAn4cQGGNhERKdylc9f25FthS90o2vHi9TokGnSo9BHcxrgoxOt1oh2bl8eJiKhVdocTZ60Nol727Qjd6fe9Arti1ElRAxsAdBEapJpNPrelmpNFvYucnTYREfkk1R3SbRIEmHbEewydu/I+1A1YEbQScjL6ADg/h22tbYQxLgqp5mT3uFgY2kRE5JNUd0i3Ju7rBxB1er3HmCXjLKCJCmodGrUauZlmTErvHdTvafPyOBEReZHyDmmfXA6Yths8Aru+5xPnn2oW5MC+mC5CgxRjTFACG2CnTUREF7nwhK8mh1OyO6QvlXDgVkRUf+oxZhlTBajDL8LC74yJiMiLr/lrXaQajU0ur/eKfYe0W3MdTLuu8BhqvGwKaq/7i/jHlimGNhER+Zy/bonYd0gDQOKeftDYT3uMWTJrAJVK1OPKHee0iYjCXGvz11GRGiTG6aBWAUmGKGQO7irqHdKqpgqYths8Aruhx2Pn567DPLABdtpERGGvtSd8NTmcmHvPIERq1aLfIX3pd64BZS6fKSZ22kREYe7CE758McZFwZQQLeod0upzx70Cu7bfCwxsH9hpExGFuQtP+Lp4TvsCseev2V23D0ObiIiC/oQvTe1XSPw0zWPMdt3bsF82UZTjhQqGNhERBfUJX+yuO45z2kRE5CbmE74iqnZ7BXb1oL8zsNuBnTYREYmO3XVgsNMmIiLR6M5s8grsqmF7GdgdxE6biIhEwe468NhpExFRQEX/uNIrsCvTvmo1sO0OJ85aG4K/epjCsNMmIqLAWauC/pKh1sLa10IlqWYTcjL6QKNmX3kphjYREXVabPF8xJx42WOsIv0HCJFJrX7O10IlF17nZpoDX6jC8c8YIiLqOMEF03aDR2A7dVfCkmVrM7BbW6ikqLiCl8p9EK3TdjgcmDNnDsrKyqBWq7Fo0SJotVrMmTMHKpUKffv2xcKFC6Hm5Q8iIkWK++o+RJVv9hycUocqq/ca3L60tlCJtbYRNXV2pBhjOltmSBEttHfv3o3m5masX78ee/fuxR//+Ec4HA7MmjULQ4cOxYIFC7Bz505kZWWJVQIREYnB1QTTzmSPoaaEW1Bz079g0sYCqPVrNxcWKvG1drcxLgrxet+LmIQz0drcnj17wul0wuVyoa6uDlqtFocPH8aQIUMAACNHjsS+ffvEOjwREYkg4bMxXoFtGVOJmpv+1e59XVioxBexFypRKtE67ZiYGJSVlWHcuHGwWq3Iz8/HgQMHoPppEfPY2FjU1rb+15jRGAOttnO/NJMprlOflwueh7zwPOSF5xEEjjrg/Uvq6zkNuPkdXBq77TmPh6ekIiY6Ep9+fRoV1eeQnBCNYddejunjr4FGI/30qdx+J6KF9ttvv420tDQ8/vjjOH36NO699144HA739vr6ehgM3l+8v5jV2tCpGkymOFgs/l2mkTOeh7zwPOSF5yE+nw9JyawBVCrgkpo7ch6339ID44Z081iopKqqvlM1B4JUv5PW/lAQ7c8Yg8GAuLjzB46Pj0dzczMGDBiAwsJCAMCePXswePBgsQ5PRESdpG4s9Qrs+p5PnP/e9U9XTQNFzIVKQolonfZ9992HuXPnIjc3Fw6HA4899hiuvfZazJ8/Hy+++CJ69eqFsWPHinV4IiLqBD6CVJ5EC+3Y2Fi8/PLLXuOrV68W65BERNRJ2urPYDyQ6THWcNVDqO/3vEQV0cX4RDQiIgLA7loJpL81j4iIJBVZvtUrsOv6LmJgyxA7bSKiMMbuWlnYaRMRhaHoE696BXbt1S8ysGWOnTYRURDZHU6P7yNLgd21cjG0iYiCQA7rRuuP/j9El/7FY6xm4AY0mcYF5fjUeQxtIqIgkHrdaHbXoYFz2kREIpNy3ej4/473Cmzr0N0MbIVip01EJDJ/1o3uKsJxQ7G7lsM9AVJiaBMRiSzY60Yn7e4NdZNnZ1+Zdgiu6O4BPU4wyeGeADkInzMlIpJI0NaNFpwwbTd4BbYly6bowAZ+vieg0maHgJ/vCdhQUCJ1aUHFTpuIKAhyMvoAOD+Hba1thDEuCqnmZPd4Z/m6FF4x6gSECGNA9i+ltu4JmJTeO2wulTO0iYiCQKNWIzfTjEnpvQM7J+tsgKngMq9hpc9dX8yfewJSjDFBrkoaDG0ioiC6sG50IPi80WyMBVAHdo5casG+J0DOOKdNRKQwqiZLy3eGh1hgA0G8J0AB2GkTESmIz7DOrAFUKgmqCR6x7wlQCoY2EZECaOq/Q+K+QR5jzTF9YL3loEQVBZdo9wQoDEObiEjmQvEhKR0VyHsClIhz2kREMqW17vcKbHvKr8I2sImdNhGRLLG7Jl/YaRMRyYjuzGavwG7o8f8Y2ASAnTYRkWywu6a2sNMmIpJY9PE/Ams9v7JV2/9lBjZ5YadNRCQhdtfUHuy0iYgkoD/yqFdg16RuZGBTq9hpExEFma/uGrkCmiy1wS+GFIWhTUQUJPEHxiGyeq/HmHXox2g23ADfT9Ym8sTQJiISmyDAtCPea5iXwqm9GNpERCJK2tUN6uYaj7HKtMNwRXeTqCJSMoY2EZEYBCdMO4xew+yuqTMY2kREAebrRrOKUSchRHhfIidqD4Y2EVGgOOthKrjca5jdNQUKQ5uIKAB8PiRlTAWgjpSgGgpVfLgKEVEnqOzlLT/VjIFNAcZOm4hCit3hRE2dHfF6HXQRGlGP5TOsM2sAlcrHu4k6j6FNRCHB6XJhQ0EJiootqLLZkWjQIdVsQk5GH2jUgb2oqKn7Fon7b/IYa9YPgPXmTwN6HKJLMbSJKCRsKCjBjs9L3a8rbXb369xMc8COwwU+SEqc0yYixbM7nCgqtvjcVlRcAbvD2eljRFR94hXYjV0mMrApqNhpE5Hi1dTZUWWz+9xmrW1ETZ0dKcaYDu+f3TXJBTttIlK8eL0OiQadz23GuCjE631va4vu9HtegV3fczYDmyTDTpuIFE8XoUGq2eQxp31Bqjm5Q3eRs7smOWKnTUQhISejDzIHd0WSIQpqFZBkiELm4K7IyejTrv1E//AHr8CuHfAnBjbJAjttIgoJGrUauZlmTErv3eHvabO7JrljaBNRSNFFaNp905n+8EOIPrXaY6z6xi1wJI0JZGlEncbQJqKwxu6alIShTURhKeGzMYioOeAxVjVsL5xx10lUEVHbGNpEFF4EAaYd3utas7smJWBoE1HYSC64HCpnvcdY5YijcEVdKVFFRO3D0Cai0Odqhmlnotcwu2tSGoY2EYU0XzeaVYwuhaD1HieSO4Y2EYWm5jqYdl3hNczumpSMoU1EIcfn17jGVALqCAmqIQocPsaUiEKG2n7GK7AFqM931wxsCgHstIkoJPjsrjNrAJVKgmqIxMFOm4iUrfqwV2A74m44310zsCnEsNMmIsXiI0gp3LDTJiLFiaja7RXYjZdNYWBTyGOnTUSisDucHV4iszXsrimciRrar732GgoKCuBwODB16lQMGTIEc+bMgUqlQt++fbFw4UKo1Wz2iUKJ0+XChoISFBVbUGWzI9GgQ6rZhJyMPtB04t+77tQ6GA7/1mOsvlceYoc9B1hqO1s2kSKIlpiFhYUoKirCunXrsGrVKpw5cwZLlizBrFmzsHbtWgiCgJ07d4p1eCKSyIaCEuz4vBSVNjsEAJU2O3Z8XooNBSUd3qdpu8ErsC1ZNjT0zutktUTKIlpof/LJJzCbzZg5cyYefPBBjBo1CocPH8aQIUMAACNHjsS+ffvEOjwRScDucKKo2OJzW1FxBewOZ7v2F3NsqdflcNs1f+blcApbol0et1qtOHXqFPLz81FaWooZM2ZAEASofvoKRmxsLGprW7+kZTTGQKvt3FyYyRTXqc/LBc9DXngevp2uqEdVrd3nNmttIzSRETAlx/q3s7U+vq6VK8DXE8P5+5CXUDkPQH7nIlpoJyQkoFevXoiMjESvXr2g0+lw5swZ9/b6+noYDK0/sN9qbehUDSZTHCwhMNfF85AXnkfLnA4nEuN0qLR5B7cxLgrOJkebx4z7+gFEnV7vMVZ94zY4kkb5nLvm70NeQuU8AOnOpbU/FES7PD5o0CB8/PHHEAQB5eXlOHfuHG6++WYUFhYCAPbs2YPBgweLdXgikoAuQoNUs8nntlRzcpt3kZu2G7wC25JlOx/YRCRepz169GgcOHAAkydPhiAIWLBgAbp27Yr58+fjxRdfRK9evTB27FixDk9EEsnJ6APg/By2tbYRxrgopJqT3eO+JBSmI8JW5DFWdfOncOoHiForkdKI+pWvJ5980mts9erVYh6SiCSmUauRm2nGpPTebX9PWxBg2hHvNcwbzYh848NViEgUuggNUowxLW5P3pEIldDsMVY54lu4oi4XuzQixWJoE1FwuRww7UzyGmZ3TdQ2hjYRBY2vR5BWjC6DoJXX12qI5Mrvu8dLS0vx0Ucfwel04uTJk2LWREQhRtVc2+IzwxnYRP7zK7T/+c9/YsaMGVi8eDGqq6tx1113YevWrWLXRkQhwLTdgORdV3qMWcZU8XI4UQf4FdpvvPEG1q1bB71ej6SkJGzZsgWvv/662LURkYKpG095ddeCWnc+rNWcmSPqCL/+5ajVauj1evfrlJQUrs5FRC3yeSk8swZQ+Xg0KRH5za/Q7tu3L1avXo3m5mYcPXoUa9euxdVXXy12bUSkMNqaz2H8LMNjzGEYhOqhuySqiCi0+NUuL1iwAOXl5dDpdJg7dy70ej0WLlwodm1EpCCm7QavwLZk2RjYRAHkV6e9aNEiLFmyBI8//rjY9RCRwkSWb0P8V7/2GGsypqFm8D8lqogodPkV2sXFxaivr0dsrJ9L6hFRWGjpa1xEJA6/b0QbPXo0evbsCZ1O5x5/9913RSuMiNrP7nC2/bzvAIg+vgL67+Z5jJ3rdj/qrv6DaMckIj9De/bs2WLXQUSd4HS5sKGgBEXFFlTZ7Eg06JBqNiEnow80Af6mB7trIun49a95yJAhOHfuHHbt2oXt27fDZrNhyJAhYtdGRH7aUFCCHZ+XotJmhwCg0mbHjs9LsaGgJGDH0B95xCuwa/u9wMAmCiK/Ou033ngD//nPfzB+/HgIgoD8/Hx89913mDFjhtj1EVEb7A4niootPrcVFVdgUnrvTl8qZ3dNJA9+hfa2bdvw/vvvIyoqCgAwZcoUTJw4kaFNJAM1dXZU2ew+t1lrG1FTZ291iczWJHyWiYiazzyPN3ADmkzjOrQ/Iuocv0JbEAR3YAOATqeDVsvHEBLJQbxeh0SDDpU+gtsYF4V4vc7Hp9rG7ppIfvxK3mHDhuGRRx7BHXfcAQDYsmULhg4dKmphROQfXYQGqWYTdnxe6rUt1Zzc7kvjyTsSoRKaPcasQ/eg2TCwU3USUef5FdpPP/001q1bhw8++ACCIGDYsGHIyckRuzYi8lNORh8A5+ewrbWNMMZFIdWc7B73iyDAtCPea5jdNZF8+BXaDQ0NEAQBK1asQHl5OdavXw+Hw8FL5EQyoVGrkZtpxqT03h36nravS+GVI47AFdU1kGUSUSf59ZWvxx9/HGfPngUAxMbGwuVy4cknnxS1MCJqP12EBinGGP8D22Vvce6agU0kP361yqdOnUJ+fj4AQK/X47HHHsOECRNELYyIRLZWBdMlQxWjSyFovUOciOTBr05bpVLh22+/db/+/vvveWmcSKFUjqoWu2sGNpG8+ZW8Tz31FKZPn44uXbpApVKhqqoKy5YtE7s2Igown2E9phJQR0hQDRG1V5ud9q5du9CtWzfs2rULv/jFLxAbG4tx48bhhhtuCEZ9RBQA6oZjLX/vmvejZ00AABgXSURBVIFNpBithvabb76JV199FXa7HceOHcOrr76K8ePHo7GxES+88EKwaiSiTjBtNyBpr+d3rC2ZNUCuIFFFRNRRrV4e37p1KzZs2IDo6GgsX74cGRkZuPPOOyEIAn7xi18Eq0Yi6gBtdSGMB7I8xlwRiagcdVyagoio01oNbZVKhejoaABAYWEhcnNz3eNEJF98BClRaGr18rhGo4HNZsOZM2dw9OhR3HLLLQCAsrIy3j1OJEO6M5u9ArspcTQDmyhEtJq8DzzwAG6//XY0Nzdj8uTJSElJwT//+U+89NJLmDlzZrBqJCI/sLsmCn2thnZ2djZSU1NhtVpx9dVXAzj/RLTFixdzwRAimYj+4UXoS57xGGu46iHU93temoKISDRtXuPu0qULunTp4n6dnp4uakFE5D9210Thxa8nohGRvMQdnuEV2LVX/4GBTRTieDcZkcKwuyYKXwxtIoVI+HQkImq/8BirGfg+mkxjA7J/u8PZoWU9iSh4GNpECiBmd+10ubChoARFxRZU2exINOiQajYhJ6MPNGrOoBHJCUObSMZ8hXXVsE/gjLs+YMfYUFCCHZ+Xul9X2uzu17mZ5oAdh4g6j39GE8mRILTYXQcysO0OJ4qKLT63FRVXwO5wBuxYRNR57LSJZMZXWFeO+AauqCsCfqyaOjuqbHaf26y1jaipsyPFGBPw4xJRx7DTJpILl73F7lqMwAaAeL0OiQadz23GuCjE631vIyJpsNMmkgFfYV0xugyCNk7U4+oiNEg1mzzmtC9INSfzLnIimWFoE0lI5ahC8kc9vMaD+b3rnIw+AM7PYVtrG2GMi0KqOdk9TkTywdAmkojPS+GZVkAV3O5Wo1YjN9OMSem9+T1tIpnjnDZRkKkbfmj5e9dBDuyL6SI0SDHGMLCJZIydNlEQ+e6uawCVSoJqiEhp2GkTBYG25oBXYDfrB/zUXTOwicg/7LSJRMYFPogoUNhpE4kksnybV2A3XjaZgU1EHcZOm0gE7K6JSAzstIkCKPrEq16BXd9rDgObiAKCnTZRgLC7JiKxsdMm2bM7nDhrbZDtilP6o495BbbtmnwGNhEFHDttki2ny4UNBSUoKragymZHokGHVLMJORl9oFHL4+9NdtdEFEwMbZKtDQUlHgtZVNrs7te5mWapygIAxB8Yh8jqvR5j1YP+DkfiSIkqIqJwII92hegSdocTRcUWn9uKiiukvVS+VuUV2JYsW8ACW+7TAUQkHXbaJEs1dXZU2ew+t1lrG1FTZ0eKMSaoNSXt6gp1s+el76qbP4NTf3VA9q+E6QAikhZDm2QpXq9DokGHSh/BbYyLQrxeF7xiBBdMOxK8hgM9dy3n6QAikgf++U6ypIvQINVs8rkt1ZwctJWoTNsN3oF9x5mAB7aspwOISDZEDe3Kykqkp6fj+++/x4kTJzB16lTk5uZi4cKFcLlcYh6aQkBORh9kDu6KJEMU1CogyRCFzMFdkZPRR/yDOxtbvjM8ukvAD+fPdAARkWiXxx0OBxYsWICoqCgAwJIlSzBr1iwMHToUCxYswM6dO5GVlSXW4SkEaNRq5GaaMSm9N2rq7IjX64LSYfsM64wzgEa8OXRZTQcQkWyJ1mkvXboUd911F1JSUgAAhw8fxpAhQwAAI0eOxL59+8Q6NIUYXYQGKcYY0QNb1VTZcnctYmAD8pkOICJ5E6XT3rx5MxITEzFixAi8/vrrAABBEKD6ad3g2NhY1NbWinFoog7xGdaZVkAVvLC8cNm/qLgC1tpGGOOikGpODs50ABEpgkoQBCHQO7377ruhUqmgUqlw9OhR9OjRA0eOHMGRI0cAADt27MC+ffuwYMGCVvfT3OyEVssOg0Rk+xb4u4+vbOUG/J+F3xqbmmG12WE06BAVyS94ENHPRPlfhDVr1rj/+5577sEzzzyDZcuWobCwEEOHDsWePXswbNiwNvdjtTZ0qg6TKQ4Wi/I7ep6HOFp9BGkrdQbjPLQAamvOQcyjyO330VE8D3kJlfMApDsXkymuxW1B+8rXU089hVdeeQU5OTlwOBwYO3ZssA5N5CGi6hOvwHZGXcVnhhOR7Il+7W3VqlXu/169erXYhyNqFRf4ICIl48NVKCzoTq3zCmy76RcMbCJSFN7lQiGP3TURhQp22hSyYr5/ziuwG3o8zsAmIsVip00hid01EYUidtoUUuK+us8rsGsHvMrAJqKQwE6bQga7ayIKdQxtUjzj3kHQNnznMVZ941Y4kkZLVBERkTgY2qRo7K6JKJwwtEmRfIV11c2FcOr7d2h/doczqMt/EhF1BEOblEVwwbQjwWu4o9210+XChoISFBVbUGWzI9GgQ6rZhJyMPtCoeZ8mEckLQ5sUw1d3XTHyOwi6Lh3e54aCEuz4vNT9utJmd7/OzTR3eL9ERGJgK0Hy52xoce66M4FtdzhRVGzxua2ouAJ2h7PD+yYiEgM7bZI1n2GdcQbQxHR63zV1dlTZ7D63WWsbUVNnR4qx88chIgoUdtokS6qmypbvDA9AYANAvF6HRIPO5zZjXBTi9b63ERFJhZ02yY7PsM6sBlSB/RtTF6FBqtnkMad9Qao5mXeRE5HsMLRJNtQN3yNpb6rHmFN3JapGHhXtmDkZfQCcn8O21jbCGBeFVHOye5yISE4Y2iQLUj0kRaNWIzfTjEnpvfk9bSKSPc5pk6S0NQe8AtuelBX0p5rpIjRIMcYwsIlI1thpk2T4CFIiovZhp01BF3n2b16B3XDVTAY2EVEb2GlTULG7JiLqOHbaFBTRP/4ZWKvyGKvt9wIDm4ioHdhpk+jYXRMRBQY7bRJN7DdPegV2zQ3rGNhERB3ETptE4au7Rq6AJktt8IshIgoR7LTJb3aHE2etDa2ufhX/3wlegW0dUhCU7tqf+oiIlIydNrXJ6XJhQ0EJiootqLLZkWjQIdVsQk5GH2jUP//dJ9Xctb/1EREpHUOb2rShoMRjUY1Km939OjfTjMQ9/aCxn/b4TOUtX8IV01MW9RERhQq2IdQqu8OJomKLz21fFJ+FabvBK7AtWbagBXZr9RUVV/BSORGFFHba1KqaOjuqbHav8b8Nut1rrGLUcQgRicEoy62l+gDAWtuImjo7UoyBWX+biEhq7LSpVfF6HRINOvdrncruM7AtWbagBzbgXd/FjHFRiNf73kZEpETstKlVuggNUs0m7Pi81HdYZ5wFNFESVHbexfVdKtWczFW7iCikMLSpTXeN7IJHhcFe42fGVMvi7uycjD4Azs9hW2sbYYyLQqo52T1ORBQqGNrUqqSPekHtqPAYK02vhC4yAnLpYTVqNXIzzZiU3hs1dXbE63XssIkoJEnfJpEsqZosMG03eAR2fc8nYMmyQRcZIWFlLdNFaJBijGFgE1HIYqdNXrjABxGRPLHTJjd1wzGvwK69ejkDm4hIJthpEwB210RESsBOO8ypG37wXj7z+ncZ2EREMsROO4wlfdQdaofVY4xhTUQkXwztMKSp/RqJnw73GKsath/OuGskqoiIiPzB0A4zl14Kd+ouR9XIb/36rN3h5PegiYgkxNAOE1rrfhg/H+sxVpn2NVzRV7X5Wa5XTUQkDwztMHBpd+2IH4zqIQV+f57rVRMRyQPbpBAWafmXV2BXpB9rV2BzvWoiIvlgpy0jgZwzvjSs7SnjYbthTbv3w/WqiYjkg6EtA4GcM9adWgvD4Qc9xipGl0HQxnWotgvrVVf6CG6uV01EFFwMbRkIyJyx4IJpR4LH0Lmu/4u6/i92qjauV01EJB8MbYm1NWc8Kb13m/uIPvEq9MVzPcYsYyyAOjBdMNerJiKSB4a2xPyZM+7a0oddzTDtTPQYqu81Fw295wS0Rq5XTUQkDwxtiXV0zjj2u4WIOf6Sx5gl0wqoxAvTC+tVExGRNPiVL4ldmDP2xeecsfMcTNsNHoFde/WL558ZLmJgExGR9Nhpy4C/c8Zxh2cg6pTn17YsmTWAShW0WomISDoMbRloc864qRqm7UaPz9iu+yvsl00KcqVERCQlhraM+Jozjj94B1C502Ms0MtnciEQIiJlYGjLlMpejuQ9fT3Gqm/8AI6kjIAdgwuBEBEpC0Nbhoz7h0Fbd8RjLNDdNcCFQIiIlIbtlIyoG47BtN3gEdjWIQVArhDwY3EhECIi5WFoy0TMsWVI2jvQY8ySZUNz/GBRjufPQ12IiEheRLk87nA4MHfuXJSVlaGpqQkzZsxAnz59MGfOHKhUKvTt2xcLFy6EmvOmPueuq24+AKe+n6jH5UIgRETKI0pqbtu2DQkJCVi7di3eeOMNLFq0CEuWLMGsWbOwdu1aCIKAnTt3tr2jEBf73UKPwK4zPwdLlk30wAY68FAXIiKSnCiddnZ2NsaOHet+rdFocPjwYQwZMgQAMHLkSOzduxdZWVliHF721Od+RNIn13qMVYw6CSEiPqh1cCEQIiJlUQmCEPi7nH5SV1eHGTNmYMqUKVi6dCk++eQTAMD+/fuxadMmLF++vNXPNzc7odWGWMf32W+Bktd/fj30TaD3dOnqAdDY1AyrzQ6jQYeoSH6hgIhIrkT7X+jTp09j5syZyM3Nxfjx47Fs2TL3tvr6ehgMhjb3YbU2dKoGkykOFkttp/YRKJr675C4b5D7taDWoWLUCUATA7RRYzDOQwugtuYcxDyKnH4fncHzkBeeh7yEynkA0p2LyRTX4jZR5rQrKiowffp0zJ49G5MnTwYADBgwAIWFhQCAPXv2YPBgce6Klh1BgOHLaR6BbbvubVSMsZwPbCIiIj+J0mnn5+fDZrNh5cqVWLlyJQDg6aefxuLFi/Hiiy+iV69eHnPeoUpT+xUSP01zv3ZGXoaqEV8D6kgJqyIiIqUSJbTnzZuHefPmeY2vXr1ajMPJjyAg/uAERFZ95B6qSd2IpuRbpauJiIgUj3cdBZi2+lMYD/wczs36AbAO28u1romIqNPCKrRFXc1KcCKhcBQiar90D1UP/hccxlsCexwiIgpbYRHaYq9mFVG5EwkH73C/bkoYjprB/wRUfOIbEREFTliEtmirWbkcSNw7EJrGk+4h65CP0Bx/Y8f3SURE1IKQbwXFWs0q8uzfYNqZ5A5su+k2WDJrGNhERCSakO+0/VnNKsXYju9LO88haXcfqJ0/f+G+6ubP4NRf3dlSiYiIWhXynfaF1ax8ae9qVrpTa2Aq6OIO7MYr7v5pgQ8GNhERiS/kO+0Lq1ldPKd9gb+rWamaa5G860qPscq0r+CK7hGoMomIiNoU8qENdG41q6gf8xH37ZPu1w1XzUR9vyWi1UpERNSSsAhtjVqN3EwzJqX39vt72ipHFZI/6uExVjniG7iirhCxUiIiopaF/Jz2xXQRGqQYY9oM7JhjyzwCu77XXFiybAxsIiKSVFh02v5S288gaY/n97Yr0n+AEJkkUUVEREQ/C6tOuzWxxfM9Aruu3/OwZNkY2EREJBth32mrz51A0ifXeYxVjC6FoDVIVBEREZFvYd1p6w8/7BHYtgErz3fXDGwiIpKhsOy0NfXFSNw32P1aUMegYtQxQNOOJ6MREREFWXiFtiDA8NU06M5udQ/VXP8umrrcLmFRRERE/gmf0HY1w7Qz0f3SqbsSVWlfAupICYsiIiLyX9iEtqq52v3f1amb4EjOkrAaIiKi9gub0BYik2HJOAtooqQuhYiIqEPC6+5xBjYRESlYeIU2ERGRgjG0iYiIFIKhTUREpBAMbSIiIoVgaBMRESkEQ5uIiEghGNpEREQKwdAmIiJSCIY2ERGRQjC0iYiIFIKhTUREpBAqQRAEqYsgIiKitrHTJiIiUgiGNhERkUIwtImIiBSCoU1ERKQQDG0iIiKFYGgTEREphFbqAsTy5ZdfYvny5Vi1apXUpXSIw+HA3LlzUVZWhqamJsyYMQNjxoyRuqwOcTqdmDdvHn744QdoNBosWbIEV111ldRldUhlZSUmTpyIt956C71795a6nA67/fbbERcXBwDo2rUrlixZInFFHfPaa6+hoKAADocDU6dOxZ133il1Se22efNmbNmyBQBgt9tx9OhR7N27FwaDQeLK2sfhcGDOnDkoKyuDWq3GokWLFPlvpKmpCXl5eTh58iT0ej0WLFiAHj16SF2WW0iG9htvvIFt27YhOjpa6lI6bNu2bUhISMCyZctgtVpxxx13KDa0d+3aBQBYv349CgsLsWTJEvz5z3+WuKr2czgcWLBgAaKioqQupVPsdjsAKPYP2gsKCwtRVFSEdevW4dy5c3jrrbekLqlDJk6ciIkTJwIAnn32WUyaNElxgQ0Au3fvRnNzM9avX4+9e/fij3/8I1555RWpy2q39957DzExMXjvvfdw7NgxLFq0CG+++abUZbmF5OXxq666SpH/z3Kx7OxsPProo+7XGo1Gwmo6JzMzE4sWLQIAnDp1CsnJyRJX1DFLly7FXXfdhZSUFKlL6ZRvvvkG586dw/Tp0zFt2jR88cUXUpfUIZ988gnMZjNmzpyJBx98EKNGjZK6pE45dOgQSkpKkJOTI3UpHdKzZ084nU64XC7U1dVBq1VmT1hSUoKRI0cCAHr16oXvv/9e4oo8KfOn2oaxY8eitLRU6jI6JTY2FgBQV1eH3/3ud5g1a5bEFXWOVqvFU089he3bt2PFihVSl9NumzdvRmJiIkaMGIHXX39d6nI6JSoqCv/zP/+DO++8E8ePH8f999+PDz/8UHH/I2u1WnHq1Cnk5+ejtLQUM2bMwIcffgiVSiV1aR3y2muvYebMmVKX0WExMTEoKyvDuHHjYLVakZ+fL3VJHdK/f3/s2rULmZmZ+PLLL1FeXg6n0ymbxikkO+1Qcfr0aUybNg0TJkzA+PHjpS6n05YuXYp///vfmD9/PhoaGqQup102bdqEffv24Z577sHRo0fx1FNPwWKxSF1Wh/Ts2RO/+tWvoFKp0LNnTyQkJCjyXBISEpCWlobIyEj06tULOp0OVVVVUpfVITabDceOHcOwYcOkLqXD3n77baSlpeHf//43tm7dijlz5rinYpRk0qRJ0Ov1mDZtGnbt2oVrrrlGNoENMLRlq6KiAtOnT8fs2bMxefJkqcvplA8++ACvvfYaACA6OhoqlUpW/wj8sWbNGqxevRqrVq1C//79sXTpUphMJqnL6pCNGzfi+eefBwCUl5ejrq5OkecyaNAgfPzxxxAEAeXl5Th37hwSEhKkLqtDDhw4gOHDh0tdRqcYDAb3zY3x8fFobm6G0+mUuKr2O3ToEAYNGoRVq1YhMzMT3bp1k7okD8q6HhZG8vPzYbPZsHLlSqxcuRLA+RvslHgT1K233oq8vDzcfffdaG5uxty5c6HT6aQuK2xNnjwZeXl5mDp1KlQqFZ577jnFXRoHgNGjR+PAgQOYPHkyBEHAggULFPfH4AU//PADunbtKnUZnXLfffdh7ty5yM3NhcPhwGOPPYaYmBipy2q37t274+WXX8Zbb72FuLg4/N///Z/UJXngKl9EREQKwcvjRERECsHQJiIiUgiGNhERkUIwtImIiBSCoU1ERKQQDG2iEFdaWoqMjAyv8X79+klQDRF1BkObiIhIIRjaRGHM5XJh8eLFuO222/DLX/7S/Vz1wsJC3HPPPe73zZkzB5s3b0ZpaSmys7MxdepU/OY3v8E333yDKVOmYOLEiZg6dSqOHz8u0ZkQhQflPQaJiNrt7NmzmDBhgtf4unXrcPr0aWzbtg1NTU245557YDabW13W9ocffsBf/vIXdO3aFXl5efjNb36DcePGYcuWLfjiiy9ktfYwUahhaBOFgZSUFGzdutVjrF+/figsLMQdd9wBjUaD6OhojB8/Hvv37/c5B35BUlKS+5Gb6enp+P3vf4+PP/4YGRkZGD16tKjnQRTueHmcKIy5XC6P14IgwOl0QqVS4eInHDscDvd/X/z8++zsbGzZsgXXX3893n77bSxcuFD8oonCGEObKIwNGzYMH3zwAZxOJ86dO4e//e1vGDp0KIxGI06ePAm73Y7q6mr897//9fn5WbNm4dChQ7jrrrvw6KOP4siRI0E+A6LwwsvjRGEsJycHx48fx4QJE+BwODB+/HhkZWUBOH/p+7bbbsOVV16JQYMG+fz8gw8+iKeffhp/+tOfEBERgWeeeSaI1ROFH67yRUREpBC8PE5ERKQQDG0iIiKFYGgTEREpBEObiIhIIRjaRERECsHQJiIiUgiGNhERkUIwtImIiBTi/wOwgxvcnHQq1QAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "##plotting on train data\n",
    "plt.scatter(X_train,Y_train)\n",
    "plt.plot(X_train,Y0,color='orange',label=\"Prediction\")\n",
    "plt.xlabel(\"Hours\")\n",
    "plt.ylabel(\"Score\")\n",
    "plt.title(\"Prediction on training data\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test Data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[16.88414476 33.73226078 75.357018   26.79480124 60.49103328]\n"
     ]
    }
   ],
   "source": [
    "Y_pred=linreg.predict(X_test)##predicting the Scores for test data\n",
    "print(Y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([20, 27, 69, 30, 62], dtype=int64)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#now print the Y_test.\n",
    "Y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAe0AAAFlCAYAAADGV7BOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3dd3hUZf428PtMyaRnUia0YOhNRBQIPQklAipSRCkKKljAKMIqHQIKEhGVFRHBtmho+xMB3XUX39AMNaKIYggiVSCU9Jn0yczz/hEcORsIAXJyptyf69rr4jxkzrnzrHrne2YyIwkhBIiIiMjpadQOQERERNXD0iYiInIRLG0iIiIXwdImIiJyESxtIiIiF8HSJiIichEsbaJqOnfuHFq3bo1BgwY5/vfQQw9hw4YNt33u5557Dhs3bgQADBo0CGaz+bpfa7FYMGbMGMfxjb6+ts2ePRu//vrrLT9+2bJl2Lp16w2/bsuWLRg9enSNnY/IFejUDkDkSry9vfHVV185ji9duoQHH3wQbdu2RatWrWrkGlef/1ry8/Nx+PDhan99bdu7dy+GDx9+y49PTU1Fs2bNaixPTZ+PSE0sbaLbUKdOHURGRuL06dM4cuQINmzYgOLiYvj7+yMpKQlffPEF1q1bB7vdDqPRiDlz5qBp06a4dOkSpk+fjsuXL6N+/frIzs52nLNly5bYt28fQkJCsHLlSmzatAk6nQ6RkZF44403MGPGDJSUlGDQoEHYuHEj2rRp4/j6999/H9988w20Wi0aN26MOXPmwGQyYfTo0Wjfvj0OHjyICxcuoGvXrpg/fz40GvnNtosXL2LevHk4f/48hBAYPHgwnn76aZw7dw5PPvkkYmJi8PPPP8NsNmPKlCmIi4uTPX7JkiW4fPkyXnnlFbz55pto0qQJXn/9dRw7dgxWqxVdu3bF1KlTodPpsHTpUiQnJ0Ov1yM4OBiJiYlITk7Gr7/+ijfffBNarbbS+d99913861//gtFoRGRkpGP91KlTeO2111BYWIjMzEy0atUKf//737FhwwbZ+Zo1a3bNrzMYDAr800GkAEFE1XL27FnRvn172drBgwdFp06dREZGhvjyyy9Fp06dhMViEUIIkZqaKkaNGiWKioqEEELs2rVL9O/fXwghxPPPPy+WLFkihBDi9OnTon379uLLL78UQgjRokULkZ2dLbZu3Sruu+8+kZeXJ4QQYuHChWL58uWVcvz59Rs2bBDDhw8XhYWFQgghli5dKsaOHSuEEOLxxx8XEydOFDabTVgsFtGjRw+xb9++St/jY489Jj799FMhhBBms1kMHDhQ/Pvf/xZnz54VLVq0ENu3bxdCCLFlyxYRGxt7zX3q1auX+OWXX4QQQkyfPl18/vnnQgghysvLxSuvvCI+/PBDkZGRIe69915RWloqhBDik08+EcnJyY6s//3vfyudNzk5Wdx///3CYrEIq9Uqnn32WfH4448LIYR44403xObNm4UQQpSVlYkHH3xQbNmypdL5qvo6IlfASZvoJvw54QKAzWZDcHAwFi9ejHr16gGomJL9/f0BADt37sSZM2cwYsQIx+PNZjPy8vKwd+9eTJs2DQAQGRmJzp07V7rWvn370L9/fwQFBQEAZsyYAaDiufVrSUlJwdChQ+Hr6wsAGDNmDFasWIGysjIAQK9evaDRaODv74/IyEjk5+fLHl9UVISDBw/i008/BQAEBARg6NChSElJwd133w29Xo+YmBgAQJs2bZCXl3fD/dq5cycOHz7seN6/pKQEQMUdilatWmHIkCGIjo5GdHQ0unbtWuW59u3bh7i4OMf+Pvzww0hKSgIATJkyBXv27MFHH32E06dP4/LlyygqKqp0jup+HZGzYmkT3YT/fU77f/1ZmABgt9sxaNAgTJkyxXF8+fJlBAUFQZIkiKve9l+nq/yvolarhSRJjmOz2VzlC87sdrvs6+12O8rLy2XZ//S/1//z66+19uc59Hq943b61depit1ux7vvvoumTZs6vgdJkqDRaLB69WocPnwY+/btw8KFC9GzZ09MnTq1yvNdnU+r1Tr+/Le//Q02mw0DBgxAbGwsLly4UOl7uZmvI3JWfPU4kUJ69OiBb775BpcvXwYArFu3Dk888QQAoGfPnvjnP/8JAMjIyEBqamqlx3fr1g3JyckoKCgAALz33ntYtWoVdDodbDZbpbLp2bMnvvzyS8fkmJSUhE6dOsHLy6taef39/XH33XdjzZo1ACpepb5582Z069btpr5vrVbrKPoePXpg1apVEEKgrKwMEyZMwOrVq3H06FE8+OCDaNq0KZ577jk8+eSTjhfXXf34q0VHR2PLli0wm82w2+2yH552796N+Ph43H///QCAn3/+GTabrdL5qvo6IlfASZtIIT169MAzzzyDsWPHQpIk+Pv7Y9myZZAkCXPnzsWMGTMwYMAA1K1b95qvPI+JicHx48cxcuRIAECzZs0wf/58+Pj4oF27dnjggQccBQsAw4YNw4ULF/DII4/AbrcjMjISb7311k1lfuutt/Daa69h48aNKCsrw8CBAzF06FCcP3++2ueIi4vDlClTMG/ePMyaNQuvv/46Bg4cCKvVim7duuHpp5+GXq/HgAED8PDDD8PX1xfe3t6YPXs2AKB379545513YLVaMWTIENl+/Pbbb3j44YcRGBiIVq1aITc3FwAwefJkxMfHw9fXF/7+/ujUqRP++OOPSuer6uuIXIEkeG+IiIjIJfD2OBERkYtgaRMREbkIljYREZGLYGkTERG5CJY2ERGRi3DqX/nKzLSoHUExwcG+yM3lOzFdjXtSGfdEjvtRGfekMlffE5Mp4Lp/x0lbJTqd9sZf5GG4J5VxT+S4H5VxTypz5z1haRMREbkIljYREZGLYGkTERG5CJY2ERGRi2BpExERuQiWNhERkYtgaRMREbkIp35zFWd18OAPSEiYgUaNGkOSJJSWluK++/pj2LARN3WeDz54D5GRjdC8eQvs3p2Cp5565ppf9913O3DnnW0hSRL+8Y+P8cor02vi2yAiIhfD0r5FHTp0xKuvJgIAysrKMGrUw+jX7wEEBFz/nWyup3nzlmjevOV1//6LL9ahUaOZiIxsxMImIvJgLl3afsdmw3Bpc42es7TOYBS2WHBTjykqKoJGo8GkSc+jXr36sFgsWLz473j77Tdw7txZ2O12PPPMBNx7b0fs3LkNn332CcLDTSgsLEZkZCMcPPgDvvrqS7z6aiL+/e/N2LTpS9jtNvToEYPWre/E8ePHsGBBAubMmY8FC+biww9X4cCB/fjwww9gMBgQGBiEGTMS8Pvvv2HNms+h1+tw4UIGeveOwxNPjKvR/SEiIvW4dGmr6ccff8ALLzwLjUYDnU6HyZOnYM2azxEX1x8xMb2wadMGBAUZMWNGAvLz8xAf/yxWr/4/LF++FB999BmaNo3Ak0+OlZ0zNzcHq1d/hs8+Wwe93gvLli1B+/b3olmzFpgyZSb0ej0AQAiBN99ciOXLP4bJFI7/+791+OyzT9CtWw9cunQBq1atg9VqxeDB/VnaREQK0ZScR8Cvz8HS9iPYvevVyjVdurQLWyy46am4plx9e/xPa9Z8jjvuiAQAnDhxHL/88hOOHPkVAGCzlSMnJxt+fn4ICjJCkiS0bdtO9vjz58+jceOmMBi8AQATJ758zWvn5eXB19cPJlM4AKB9+3uwcuVydOvWA02aNINOp4NOp3Och4iIapbfsdnwPbP0yp9nwtLuH7VyXZcubWek0VS8ID8yshHCw8MxZsxYlJaW4LPPPkVAQCAKCgqRm5sLkykAR48eQXh4HcdjGzSIwB9/nEZZWRm8vLwwe/ZUvPTSK9BoNLDb7Y6vMxqNKCoqRFZWFsLCwnDo0EE0bHgHAECSavf7JSLyJJqSDITuaiVbs7RdUWvXZ2krZNCgoVi0aAFeeOFZFBYWYMiQR6DX6zFzZgJefvkFhIaGwG6XN2xwcDAee+wJvPDCs5AkCd2794TJFI62bdthwYK5mDp1FgBAkiRMnToLs2ZNgUYjISAgEDNnzsPJk8fV+FaJiDyC3+/z4Hv6Hcexpc37KGkwulYzSEIIUatXvAnu/HnaJlOAW39/t4J7Uhn3RI77URn3pLKa3hNN6UWEprSQrWX2ygB0/jV2javx87SJiIhuge/x+bLCtrR+F5lxZsUK+0Z4e5yIiOh/SKWXEZbSTLaW1eschC5QpUQVOGkTERFdxffEG7LCtrR6G5lxZtULG+CkTUREBACQyrIQ9l0T2VpW7FkIfZBKiSrjpE1ERB7P9+RiWWFbWr5ZMV07UWEDnLSJiMiDSWXZCPuusWwtK/YPCL1RpURV46RNREQeyefUEllhF7RYeGW6ds7CBjhpExGRh5GsOQjb2Ui2lhV7GkIfok6gm8BJm4iIPIbPmWWywi5oPv/KdO38hQ1w0iYiIg8gWXMRtjNStpYVcwrCK1SlRLeGkzYREbk1nz+Wywq7oNnciunaxQob4KRNRERuSrLmI2xnQ9laVsxJCK8wlRLdPk7aRETkdrzPfiQr7MKms65M165b2AAnbSIiciNSuRlYG4irPycrK+YEhJdJtUw1iZM2ERG5Be9znyJsR4TjuLDJtCvTtXsUNsBJm4iIXJxUbkHYjgaytazo3yEMdVRKpBxO2kRE5LK8z30mK+zCxq8Ao4RbFjbASZuIiFyRrRCm7fVkS9k9f4Pdux78VIpUGxQr7Y0bN2LTpk0AgNLSUqSnp2Pt2rVYuHAhJElC8+bNMXfuXGg0HPaJiKj6DBlrEJg2wXFc1GgSCpu/pmKi2iMJIYTSF3n11VfRqlUr7NixA0899RQ6d+6MhIQE9OzZE3Fxcdd9XGamReloqjGZAtz6+7sV3JPKuCdy3I/KPGpPbEUI214PEv6qreye6bB7y5/PdvU9MZkCrvt3io+5hw8fxvHjxzF8+HCkpaUhKioKABAdHY29e/cqfXkiInIDhox1MG2v6yjsojvikRlnrlTY7k7x57RXrlyJ+Ph4AIAQApIkAQD8/PxgsVT9k1BwsC90Oq3SEVVT1U9Tnop7Uhn3RI77UZlb70l5MbDBCNjL/lobdAa+fnfAt4qHueueKFraZrMZJ0+eRJcuXQBA9vx1YWEhAgMDq3x8bm6RkvFU5eq3b5TAPamMeyLH/ajMnffEcOELBP46znFc1PA5FLZaDBQBKLr+91zbe1JSUgJvb+8aO59qt8cPHDiAbt26OY7btGmD1NRUAEBKSgo6duyo5OWJiMgV2UoQtq2urLCzexyuKGwnNG7c6Fq7lqKlferUKURE/PXuNNOmTcN7772H4cOHw2q1ol+/fkpenoiIXIzh4kaYtodDslfcaS2OGFfx3LVP5A0eqY60tF+xf/8+pKcfqZXrKXp7/Omnn5YdN27cGKtXr1bykkRE5IrspQj9rhk05fmOpezuP8Pu21jFUNc3efILyMg4j7y8PFgsZkyaFA+j0Yj69RtgyZJlil2XvyRNRESq8rr0NUzbTI7CLm7wZMV07aSFDQCJiW8hIqIh0tPTAADp6WmIiLgDiYlvKXpdviMaERGpw16G0JQW0FhzHEvZ3X+C3bepiqGqx9vbGxMmvIiNGzfAYDBAp9MhPn5ijb4g7Vo4aRMRUa3zuvwNTNvCHIVdUm/Ulena+Qv7T2vWfIaoqCisX78RHTt2QlLSPxS/JidtIiKqPXYrQnbdCW3ZRcdSTrcfYfNrrmKoWxMb2wezZ78KrVaLLl26YffuFMWvyUmbiIhqhVfmf2HaFuoo7JK6jyIzzuyShQ0AMTG9oNVWvAGYVqtFTEwvxa/JSZuIiJRltyJkT3toS846lnK6HoDNv6WKoVwTS5uIiBTjlfX/EPTTMMdxSZ2hsLRbpV4gF8fSJiKimmcvR8jee6EtPu1YyumaCpt/a/UyuQGWNhER1Sh99nYYDw52HJeGD4L57iQVE7kPljYREdUMYUPw3k7QFR13LOV02QdbwJ0qhnIvLG0iIrpt+uydMB58yHFcarof5rvXAVc+jplqBkubiIhunbAheH936Ar++sCMnC67YQtop2Io98XSJiKiW6LP2Q3jj/c7jktD42C+ZwOnawWxtImI6OYIO4yp0dBbfnEs5XZOQXlgexVDeQaWNhERVZs+dy+MP/R3HJeF9EL+vZs5XdcSljYREd2YsMP4fW/ozQcdS7lRO1Ae1EHFUJ6HpU1ERFXS5aUi+ECc47gsuCfyO/yb07UKWNpERHRtQsB44D7o81MdS7lR21Ae1EnFUJ6NpU1ERJXo8g8g+Ps+jmOrsQvyOn7L6VplLG0iIvqLEAj64X545e1xLOV2Ska5sbOKoehPLG0iIgIA6PJ/RPD3f30mtDWwA/KitgGSRsVUdDWWNhGRpxMCQQcHwytnh2Mpr+MWWIO7qRiKroWlTUTkwXTmQwhOjXYcWwPaI6/zTk7XToqlTUTkiYRA0E8Pwyt7q2Mpr8M3sIb0VDEU3QhLm4jIw2gtvyBkfw/Hcbl/G+R22QNIWhVTUXWwtImIPIUQCDw0Aoas/zqW8u79GtbQWPUy0U1haRMReQCtJQ0h+7s6jsv9WiC3ayqnaxfD0iYicmdCIPCXx2G4/C/HUt69m2AN7VPFg8hZsbSJiJxcSUkJvL29b/px2oJ0hOz7601RbD6NkdPtR0DD//S7Kr6mn4jIyY0bN/qmHxPwy5Oyws6/ZwNyevzMwnZx/H+PiMiJpaX9iv379yE9/Qhat25zw6/XFh5DyN6OjmObd0PkdD8EaPRKxqRawtImInJCkye/gIyM88jLy4PFYsakSfEwGo2oX78BlixZds3HBBweB++LXziO89v/E2WmAbUVmWoBb48TETmhxMS3EBHREOnpaQCA9PQ0RETcgcTEtyp9rbbwd5iSAx2FbfOqi8w+2SxsN8TSJiJyQt7e3pgw4UVotToYDAbodDrEx0+s/IK0/U8hZG8Hx2H+3euQE3OMt8PdFEubiMhJrVnzGaKiorB+/UZ07NgJSUn/cPydpugETMmBwMlVAAC7PhSZfbJQFv6ASmmpNvA5bSIiJxUb2wezZ78KrVaLLl26YffuFACA/5EX4XP+M8fX5bdbjbI6D6kVk2oRS5uIyEnFxPz12dZarRa9OjVCaHKgY82uM0Lz8EWU5ZSpEY9UwNvjREQuwD99MkL33O04Nt+1Ctm9/gC0BhVTUW3jpE1E5MS0lsMI2d/dcSy0fsiKOQVob/4d0sj1KVraK1euxPbt22G1WjFy5EhERUVh+vTpkCQJzZs3x9y5c6HRcNgnIrqW4D33QFd0wnFsbvsJSus9omIiUptijZmamoqffvoJ69atQ1JSEi5evIjExERMmjQJa9euhRAC27ZtU+ryREQuS2tJgyk5UFbYmb0vsbBJudLevXs3WrRogfj4eIwfPx6xsbFIS0tDVFQUACA6Ohp79+5V6vJERC4peG9n2UdoFkW+hMw4M6D1UTEVOQvFbo/n5uYiIyMDK1aswLlz5zBhwgQIISBJEgDAz88PFoulynMEB/tCp3Pfz3o1mQLUjuB0uCeVcU/k3HY/8o8A39wpXxuWB1+vIPje4KFuuye3wV33RLHSNhqNaNKkCby8vNCkSRMYDAZcvHjR8feFhYUIDAys4gxAbm6RUvFUZzIFIDOz6h9aPA33pDLuiZy77odxfzT0lkOO46I7JqCw5SIgHwCq/n7ddU9uh6vvSVU/cCh2e7xDhw7YtWsXhBC4dOkSiouL0bVrV6SmpgIAUlJS0LFjxxuchYjIff35nuFXF3ZW7JmKwia6BsUm7V69euHAgQMYNmwYhBBISEhAREQE5syZg3feeQdNmjRBv379lLo8EZFTM37fB/r8A47j4oinUdD6HRUTkStQ9Fe+pk6dWmlt9erVSl6SiMipaYpOIHTPPbK1rJhTEF6hKiUiV8I3VyEiqiVBBwbAK2+P47i4wZMoaLNUxUTkaljaREQK0xSfRujudrK1rJiTEF5hKiUiV8XSJiJSUNCPg+CVs8NxXFJvFCxtV6iYiFwZS5uISAGa4j8QurutbC0r+jiEIVylROQOWNpERDUs8KdhMGT9P8dxSd1hsNz1qYqJyF2wtImIaoim5BxCd7WRrWVHH4PdUFelRORuWNpERDUg8NAoGDL/7TguDR8E891JKiYid8TSJiK6DZqSCwjd1VK2lt3zKOze9VVKRO6MH2ZNRHSLAn55UlbYpab7kRlnZmGTYjhpExHdJKn0EsJSmsvWsnukwe7TUKVE5Ck4aRMR3YSAw0/LCrsstG/FdM3CplrASZuIqBqkskyEfddUtpbd4zDsPpEqJSJPxEmbiOgG/NOelxV2WUjMlemahU21i5M2EdF1SGXZCPuusWwtu/sh2H2bqJSIPB0nbSKia/A/MklW2FZj14rpmoVNKuKkTUR0Fcmag7CdjWRrOd0OwubXTJ1ARFfhpE1EdIX/0VdkhW0NvBeZcWYWNjkNTtpE5PEkax7Cdt4hW8vp9gNsfi1USkR0bZy0icij+f02U1bY5f53XZmuWdjkfDhpE5FHkqz5CNspf0OUnK6psPm3VikR0Y1x0iYij+P3+1xZYZf7tayYrlnY5OQ4aRORx5DKLQjb0UC2ltNlL2wBbVVKRHRzOGkTkUfwPT5fVtg2n0YV0zULm1wIJ20icm+2Qpi215Mt5XbehfLAu1UKRHTrOGkTkdvyPfGGrLBthvrIjDOzsMllcdImIvdjK4Jpe13ZUm7UTpQH3atSIKKawUmbiNyKz6m3ZYVt14dWTNcsbHIDnLSJyD3YimHaXke2lBu1DeVBnVQKRFTzOGkTkcvzOf2urLDt2oAr0zULm9wLJ20icl22Epi2h8uWcjslo9zYWaVARMpiaRORS/I58z78j81wHAuNAVl9MlVMRKQ8ljYRuRZ7GUzbwmRLeR3/C2twd5UCEdUeljYRuY5j78P0wwuypcw4s0phiGofS5uInJ/dCtO2UNlSXod/wRoSo1IgInWwtInIqXmf+wcC0l+SrWX2zQckSaVEROphaRORc7KXw7QtRL7W6/8hU9dFnTxEToC/p01ETsf7fFKlws7smw/Ui1MpEZFz4KRNRM7jGtN1/j0bUBZ2n0qBiJyLoqU9ePBgBAQEAAAiIiIwfvx4TJ8+HZIkoXnz5pg7dy40Gg77RAQYMtYiMG28bI3PXRPJKVbapaWlAICkpCTH2vjx4zFp0iR07twZCQkJ2LZtG+LieLuLyKMJG0xbg2VL+e3/iTLTAJUCETkvxcbco0ePori4GGPHjsWYMWNw6NAhpKWlISoqCgAQHR2NvXv3KnV5InIBhgtfVCrszL75LGyi61Bs0vb29sa4cePwyCOP4PTp03jmmWcghIB05VaXn58fLBZLlecIDvaFTqdVKqLqTKYAtSM4He5JZW65J8IOrPuff7d7bgQaDoHpBg91y/24TdyTytx1TxQr7caNGyMyMhKSJKFx48YwGo1IS0tz/H1hYSECAwOrPEdubpFS8VRnMgUgM7PqH1o8DfekMnfcE8PFjQg8/KRsLbNvHiBpgBt8r+64H7eLe1KZq+9JVT9wKHZ7fMOGDXjjjTcAAJcuXUJBQQG6d++O1NRUAEBKSgo6duyo1OWJyNkIO0zJgbLCzm/3WcXbkEp8QSpRdSg2aQ8bNgwzZszAyJEjIUkSFi5ciODgYMyZMwfvvPMOmjRpgn79+il1eSJyIl6X/4Wgnx+TrTmmayKqtmqV9h9//IFDhw5h4MCBSEhIwJEjRzBv3jzcdddd132Ml5cX3n777Urrq1evvvW0RORahIBpa5Bsydz2E5TWe0SlQESurVo/5s6YMQN2ux3btm3D6dOnMWPGDLz++utKZyMiF+Z1+T+VCjuzby4Lm+g2VKu0S0tLMXjwYOzYsQMDBw5Ex44dUVZWpnQ2InJFQsCUHIign0c4lsx3rrzy3LX7/jYIUW2oVmlrtVp8++232LlzJ2JjY7F161a+kxkRVeKV+e21p+v6I1VKROReqvWc9muvvYZVq1Zh7ty5CA8PxzfffIMFCxYonY2IXMW1nrtusxylDR5XKRCRe6pWabds2RLPP/88Tpw4AZvNhr/97W9o2LCh0tmIyAXos7fBeHCIbC2zTw6g4ecREdW0at3j/s9//oPnn38er7/+OvLy8jBixAh89dVXSmcjImd25bnrqwvb0nppxXPXLGwiRVSrtD/66COsW7cOfn5+CA0NxaZNm/Dhhx8qnY2InJQ+57vKz133yUZJxJPqBCLyENX6cVij0cDf399xHB4ezheiEXkoU7L87Yctrd5GScNnVEpD5FmqVdrNmzfH6tWrUV5ejvT0dKxduxatWrVSOhsRORF97h4Yf5B/+lZmnyxA46VSIiLPU61xOSEhAZcuXYLBYMDMmTPh7++PuXPnKp2NiJxE2NZQWWEXtHzjynPXLGyi2lStSXv+/PlITEzEyy+/rHQeInIiurz9CD5wn2wts08moDGolIjIs1WrtI8dO4bCwkL4+fkpnYeInETo9vrQ2AocxwXNF6C40UQVExFRtV+I1qtXLzRu3BgGw18/YX/++eeKBSMidejyvkfwgb6ytczelwGtt0qJiOhP1SrtKVOmKJ2DiJxA6M5G0FhzHMcFzeaiuDGfFiNyFtUq7aioKHz33XfYv38/ysvL0blzZ/Tt2/fGDyQil6DL/xHB3/eSrWX2vghofVVKRETXUu03V1m2bBnq1auHiIgIrFixAh988IHS2YioFoSktJQVdmHTWRWvDGdhEzmdak3aX3/9Nb744gt4e1c8p/Xoo49i6NChmDBhgqLhiEg5OvPPCE7tKVvL7H0B0PIFp0TOqlqlLYRwFDYAGAwG6HR8b2EiVxWy6y5oS844jgsbT0VRs9kqJiKi6qhW83bp0gUvvvgihgyp+GCATZs2oXPnzooGI6Kap7UcRsj+7rK1rF7nIXQBKiUioptRrdKeNWsW1q1bh82bN0MIgS5dumD48OFKZyOiGhS8pwN0Rb87josaTUZh81dVTEREN6tapV1UVAQhBJYuXYpLly5h/fr1sFqtvEVO5AK0BekI2Se/M5bV6xyELvA6jyAiZ/wO0fgAABnrSURBVFWtV4+//PLLuHz5MgDAz88PdrsdU6dOVTQYEd2+4H1dZYVddMcLyIwzs7CJXFS1RuWMjAysWLECAODv74/Jkydj0KBBigYjolunLfgNIfs6ydayYv+A0BtVSkRENaFak7YkSfjtt98cxydOnOCtcSInZdwfIyvsoobPVUzXLGwil1et5p02bRrGjh2LOnXqQJIk5OTkYPHixUpnI6KboC38HSF7O8jWsmJPQ+hDVEpERDXthpP2jh070LBhQ+zYsQP3338//Pz8MGDAANx99921kY+IqsH4fZyssIsbjL0yXbOwidxJlaX9ySefYNmyZSgtLcXJkyexbNkyDBw4ECUlJXjzzTdrKyMRXYem6CRMyYHQ56c61rJiTqGgzd9VTEVESqny9vhXX32Ff/7zn/Dx8cFbb72F3r1745FHHoEQAvfff39tZSSiawj64QF45e5yHBfXfxwFdy5XMRERKa3K0pYkCT4+PgCA1NRUjBo1yrFOROrQFJ9G6O52srWsmBMQXiaVEhFRbamytLVaLcxmM4qKipCeno7u3Sve/vD8+fN89TiRCoIODoZX9nbHcUm94bC0/UjFRERUm6ps3meffRaDBw9GeXk5hg0bhvDwcPznP//BkiVLEB8fX1sZiTyepvgsQnffKVvLiv4dwlBHpUREpIYqS7t///645557kJubi1atWgGoeEe0BQsW8ANDiGpJ4E+PwpC1xXFcUmcoLO1WqReIiFRzw3vcderUQZ06f/00HxMTo2ggIqqgKTkPrG0Nw1Vr2T1/g927nmqZiEhdfGKayAkF/vw4DJe/dhyXmh6Euf1aFRMRkTNgaRM5EU3JBYTuailby+6ZDrt3A5USEZEzqdZ7jxOR8gJ+eVJW2KVh/YBRgoVNRA6ctIlUJpVeRlhKM9lado802H0agr95TURX46RNpKKAX5+VFXZZSC9kxplh92moYioiclactIlUIJVlIey7JrK17B6/wO7TSJ1AROQSFJ20s7OzERMTgxMnTuDMmTMYOXIkRo0ahblz58Jutyt5aSKn5Z/2gqywy4J7XJmuG6kXiohcgmKlbbVakZCQAG9vbwBAYmIiJk2ahLVr10IIgW3btil1aSKnJJVlw5QcCJ+Mzx1r2d1/Qn7H/6iYiohciWKlvWjRIowYMQLh4eEAgLS0NERFRQEAoqOjsXfvXqUuTeR0/NMnI+y7xo5ja1BUxXTt21TFVETkahR5Tnvjxo0ICQlBz5498eGHHwIAhBCOTwfz8/ODxWK54XmCg32h02mViOgUTKYAtSM4Hbfbk7JcYEOIfO3B36APbFHtV4a73Z7cJu5HZdyTytx1TxQp7S+//BKSJGHfvn1IT0/HtGnTkJOT4/j7wsJCBAYG3vA8ublFSsRzCiZTADIzb/yDiydxtz3xOzoVvmdXOI6tAe2R1yUFKAVQze/T3fbkdnE/KuOeVObqe1LVDxyKlPaaNWscfx49ejTmzZuHxYsXIzU1FZ07d0ZKSgq6dOmixKWJVCdZ8xC28w7ZWk7XA7D5t7zOI4iIqqfWfk972rRpeO+99zB8+HBYrVb069evti5NVGv8js2WFXa5fxtkxplZ2ERUIxT/Pe2kpCTHn1evXq305YhUIZWbEbYjQraW03U/bP5tVEpERO6I74hGdJv8fp8nK+xy36ZXpmsWNhHVLL4jGtGtKi+AaUd92VJOlz2wBdylUiAicnectIluge/xBbLCtnk3rJiuWdhEpCBO2kQ3w1YI0/Z6sqXczikoD2yvUiAi8iSctImqyffkm7LCtnnVQWacmYVNRLWGkzbRjdiKYNpeV7aUG7Ud5UEdVQpERJ6KkzZRFXxOvSMrbLvOWDFds7CJSAWctImuxVYC0/Zw2VJup2SUGzurFIiIiKVNVInP6ffg//ssx7HQ+CKrz0UVExERVWBpE/3JXgrTNvlnb+V2/BblwV1VCkREJMfSJgLg88cH8P9tmuNYSFpk9c1VMRERUWUsbfJs9jKYtoXJlvI6fANrSE+VAhERXR9LmzyW99mPEHD0ZdlaZpxZpTRERDfG0ibPY7fCtC1UtpR379ewhsaqk4eIqJpY2uRRvM+tQkD6RNlaZt98QJJUSkREVH0sbfIM9nKYtoXIlvLu2QhrWF+VAhER3TyWNrk9Q8YaBKZNkK1xuiYiV8TSJvclbDBtDZYt5bf/AmWmfioFIiK6PSxtckuGC+sR+OuzsjVO10Tk6lja5F6uNV3fvQ5l4Q+oFIiIqOawtMltGC5uQODhsbI1TtdE5E5Y2uT6hB2mrUbZUn67JJTVGaRSICIiZbC0yaV5XdqMoF/GyNYy++YBEj8qnojcD0ubXNM1pmvzXf9Aad2HVQpERKQ8lja5HK/L/0bQz6Nka5yuicgTsLTJdQgB09Yg2ZK57ccorfeoSoGIiGoXS5tcglfmfxF0aLhsLbNvLiBpVUpERFT7WNrk3K41Xd/5AUrrP6ZSICIi9bC0yWnps5Jh/En+wrLMPjmAhv/YEpFn4n/9yPlcY7q2tFmGkgZjrvMAIiLPwNImp6LP3g7jwcGytcw+2YBGr1IiIiLnwdIm5yAEsFbC1b95bWm1BCUNx6kWiYjI2bC0SXX6nBQYf3xQtpbZJwvQeKmUiIjIObG0SVVhyUGQIBzHlpZvouSO8SomIiJyXixtUoU+dw+MPwyQLw4vQUlOmTqBiIhcAEubal3YNhMke6njuKDFQhRHvgCT1gCApU1EdD0sbao1urxUBB+Ik61l9r4MaL1VSkRE5FpY2oSSkhJ4eytbnKE7IqApNzuOC5q/huJGkxS9JhGRu1GstG02G2bPno1Tp05Bq9UiMTERQghMnz4dkiShefPmmDt3LjQafjKT2saNG401a75Q5Ny6/AMI/r6PbC2z9yVA66PI9YiI3Jlijbljxw4AwPr16zFx4kQkJiYiMTERkyZNwtq1ayGEwLZt25S6PFVTWtqv2L9/H9LTj9T4uUN3NpEVdmGzBGTGmVnYRES3SLFJu2/fvoiNjQUAZGRkICwsDDt37kRUVBQAIDo6Gnv27EFcXFwVZyGlTJ78AjIyziMvLw8WixmTJsXDaDSifv0GWLJk2W2dW5d/EMHfx8rWMntfBLS+t3VeIiJPp+hz2jqdDtOmTUNycjKWLl2KHTt2QJIkAICfnx8sFkuVjw8O9oVO574fvWgyBah27Y8/XomJEyciKSkJAJCenoYxY8bg3Xffvb3ntzc3BIrO/XXcdi7Qbh5M1Xy4mnvirLgnctyPyrgnlbnrnkhCCHHjL7s9mZmZePTRR1FQUIADBw4AALZu3Yq9e/ciISGhisdVXequzGQKUP37O378d8TFxaC83Aq9Xo+tW3ehSZOmt3QureUXhOzvIVvL7JUB6PyrfQ5n2BNnwz2R435Uxj2pzNX3pKofOBR7Tnvz5s1YuXIlAMDHxweSJKFt27ZITU0FAKSkpKBjx45KXZ6qYc2azxAVFYX16zeiY8dOSEr6xy2dJ2R3O1lhFzZ+peK565sobCIiujHFJu2ioiLMmDEDWVlZKC8vxzPPPIOmTZtizpw5sFqtaNKkCRYsWACt9vq3v135J6UbcYafBL/7bgd69IiGVquFzWbD7t0piInpVe3Hay1pCNnfVbaW1es8hO7Wbks5w544G+6JHPejMu5JZa6+J1VN2rVye/xWufKm34ir/0MVvLcTdIW/OY6LIl9CYYv5t3VOV98TJXBP5LgflXFPKnP1PamqtPnmKnRTtAXpCNnXWbaWFXsWQh+kUiIiIs/B0qZqC97XHbqCw47jojueR2HLN1RMRETkWVjadEPawmMI2St/0WBW7BkIfbBKiYiIPBNLm6pkTO0FvflHx3FxxNMoaP2OiomIiDwXS5uuSVt4HCF775WtZcWcgvAKVSkRERHx0zqoEuOB+2SFXdzgCWTGmZ22sEtKStSOQERUK1ja5KApOglTciD0efsda1kxJ1HQ5j0VU93YuHGj1Y5ARFQrWNoEAAj6cSBC97R3HJfUG3Vlug5TMdWNKfkpZUREzobPaXs4TfEfCN3dVraWFX0cwhCuUqLqUfJTyoiInBUnbQ/mdWmTrLBL6g6rmK6dvLABIDHxLURENER6ehqAik8pi4i4A4mJb6mcjIhIOSxtT2Qrgt9v0xH0yxOOpezoY7Dc9amKoW6Ot7c3Jkx4EVqtDgaDATqdDvHxE2/vY0WJiJwcS9vD6PL2I3h/d/j+sRzlvs2Q2ykZmXFm2A111Y5202rqU8qIiFwFn9P2FLZi+B2fD58/3gcAFEW+iMKmswGtj8rBbl1sbB/Mnv0qtFotunTpht27U9SORESkKJa2B9DlpSIgbQJ0RcdR7tMElrYrUG7sonas23b1x4hqtdqb+lhRIiJXxNJ2Z7Zi+J14HT5nlgEQFR/w0SwB0PqqnYyIiG4BS9tN6fIPVEzXhcdg82kMy50fwBrcTe1YRER0G1ja7sZWAr+TifA5/S4k2FHUcDwKm88FtH5qJyMiotvE0nYjuvwfr0zXR2HzaQRLm+WwhvRQOxYREdUQlrY7sJfC9+Qi+J5eAknYUNzwGRQ0exXQ+audjIiIahBL28XpzD9VTNcFR2DzjoTlzvdhDYlWOxYRESmApe2q7GVXput3KqbriHEoaD6f0zURkRtjabsgnfnQlek6DTbvhrC0eR/W0Fi1YxERkcJY2q7EXgbfU4vhe+ptSKIcxQ3GorDFfAhdgNrJiIioFrC0XYTW8gsCf50AXcFh2LwjYGmzDNbQ3mrHIiKiWsTSdnZ2K3xPvQXfU4uvTNdPoLDF6xC6QLWTERFRLWNpOzGt5VcEpE2A3vIzbIYGsLR5D9awvmrHIiIilbC0nZHdCt/TS+B7chEkYUVx/dEobLEQQh+kdjIiIlIRS9vJaAuOIODXCdBbfoLNUA8FrZeizNRP7VhEROQEWNrOwl4O3zN/h++JNyCJMpTUfwwFLRIh9Ea1kxERkZNgaTsBbcFRBKQ9B735J9i86qKgzVKUmfqrHYuIiJwMS1tN9nL4nHkPfider5iu641AQctFEPpgtZMREZETYmmrJT8dxgOjoTf/CJtXHRS0fhdl4fernYqIiJwYS7u2CRt8ziwDTiyA3l6KkrqPoqDVmxD6ELWTERGRk2Np1yJt4e8ISBsPff4BwDsc+S3/jrLwB9WORURELoKlXRuEDT5nlsPvxHxI9hKU1HkY3t1XoMxiUDsZERG5EJa2wiqm6+ehz0+FXR8Gc9uPUVbnIXh7BwAWi9rxiIjIhbC0lSLs8PnjA/gdf/XKdD0UBa3egvAKUzsZERG5KJa2AjRFJxCY9jz0eftg14fC3HYlyuoMUTsWERG5OI8r7ZKSEnh7eytzcmGHz9mV8Pt9HiR7MUrDB8HS+h0IL5My1yMiIo+iSGlbrVbMnDkT58+fR1lZGSZMmIBmzZph+vTpkCQJzZs3x9y5c6HRaJS4fJXGjRuNNWu+qPHzaopOISDteXjl7YFdHwLLnctRWmcoIEk1fi0iIvJMipT2119/DaPRiMWLFyM3NxdDhgxBq1atMGnSJHTu3BkJCQnYtm0b4uLilLj8daWl/Yr9+/chPf0IWrduUzMnFXZ4n/sY/scSINmLUBo+EJZWSyAM4TVzfiIioisUKe3+/fujX7+/PplKq9UiLS0NUVFRAIDo6Gjs2bOn1kp78uQXkJFxHnl5ebBYzJg0KR5GoxH16zfAkiXLbvm8muLTCEiLh1fuLtj1wbC0eQ+ldYdxuiYiIkUoUtp+fn4AgIKCAkycOBGTJk3CokWLIF0pMz8/P1iq8etOwcG+0Om0t53n449XYuLEiUhKSgIApKenYcyYMXj33Xdv7fltYQeOrwR+mgKUFwIRg6DptAKBPnVv6jQmU8DNX9vNcU8q457IcT8q455U5q57otgL0S5cuID4+HiMGjUKAwcOxOLFix1/V1hYiMDAwBueIze3qMbyPPXUeKxduw4GgwE6nQ5jx06AxWKFxWK9qfNois8g4MgL8Mr5DnadEQVtP0Rp3eFAgQQUVP/3rk2mAGRm8ve0r8Y9qYx7Isf9qIx7Upmr70lVP3Ao8kqwrKwsjB07FlOmTMGwYcMAAG3atEFqaioAICUlBR07dlTi0te1Zs1niIqKwvr1G9GxYyckJf3j5k4gBLzPfYrgfV3hlfMdSsMGILfb9yitN4K3w4mIqFYoMmmvWLECZrMZy5cvx/LlywEAs2bNwoIFC/DOO++gSZMmsue8a0NsbB/Mnv0qtFotunTpht27U6r9WE3x2SvT9Q7YdUaY71yB0nojWdZERFSrJCGEUDvE9ah+e0MIeJ//DH7HZkFjs6A07D4UtF4Ku3f92z61q9++UQL3pDLuiRz3ozLuSWWuvidV3R73uDdXqS5NyTkEHHkRXtnbYNcFwnznByitN4rTNRERqYal/b+EgHfGavgdmwFNuRlloX1hafMe7N4N1E5GREQejqV9FU1JBvyPvAhDdjLsukBY2ryPkvqPc7omIiKnwNIGACFguLAW/r9Nh6Y8H2WhvWFpswx27wi1kxERETl4fGlrSi7AP30iDFnfwq4NgKX1UpQ0eILTNREROR3PLW0hYLiwHv6/TYOmPA9lIbEV07XPHWonIyIiuiaPLG1N6UX4H3kJhqz/wq71h6XVEpREjOV0TURETs2zSlsIGC7+H/yPTrkyXcdcma4j1U5GRER0Q55T2sKOgMNj4X1pI4TWD5ZWb6MkYhwg1f5nehMREd0Kzyltewm8cnagLDi64veufRurnYiIiOimeE5pa32RHX0c0OjVTkJERHRLPOveMAubiIhcmGeVNhERkQtjaRMREbkIljYREZGLYGkTERG5CJY2ERGRi2BpExERuQiWNhERkYtgaRMREbkIljYREZGLYGkTERG5CJY2ERGRi5CEEELtEERERHRjnLSJiIhcBEubiIjIRbC0iYiIXARLm4iIyEWwtImIiFwES5uIiMhF6NQO4GmsVitmzpyJ8+fPo6ysDBMmTECfPn3UjqUqm82G2bNn49SpU9BqtUhMTMQdd9yhdizVZWdnY+jQofj000/RtGlTteOobvDgwQgICAAAREREIDExUeVE6lu5ciW2b98Oq9WKkSNH4pFHHlE7kqo2btyITZs2AQBKS0uRnp6OPXv2IDAwUOVkNYelXcu+/vprGI1GLF68GLm5uRgyZIjHl/aOHTsAAOvXr0dqaioSExPxwQcfqJxKXVarFQkJCfD29lY7ilMoLS0FACQlJamcxHmkpqbip59+wrp161BcXIxPP/1U7UiqGzp0KIYOHQoAePXVV/Hwww+7VWEDvD1e6/r374+XXnrJcazValVM4xz69u2L+fPnAwAyMjIQFhamciL1LVq0CCNGjEB4eLjaUZzC0aNHUVxcjLFjx2LMmDE4dOiQ2pFUt3v3brRo0QLx8fEYP348YmNj1Y7kNA4fPozjx49j+PDhakepcZy0a5mfnx8AoKCgABMnTsSkSZNUTuQcdDodpk2bhuTkZCxdulTtOKrauHEjQkJC0LNnT3z44Ydqx3EK3t7eGDduHB555BGcPn0azzzzDLZs2QKdznP/E5abm4uMjAysWLEC586dw4QJE7BlyxZIkqR2NNWtXLkS8fHxasdQBCdtFVy4cAFjxozBoEGDMHDgQLXjOI1Fixbh22+/xZw5c1BUVKR2HNV8+eWX2Lt3L0aPHo309HRMmzYNmZmZasdSVePGjfHQQw9BkiQ0btwYRqPR4/fEaDSiR48e8PLyQpMmTWAwGJCTk6N2LNWZzWacPHkSXbp0UTuKIljatSwrKwtjx47FlClTMGzYMLXjOIXNmzdj5cqVAAAfHx9IkuTRTxusWbMGq1evRlJSElq3bo1FixbBZDKpHUtVGzZswBtvvAEAuHTpEgoKCjx+Tzp06IBdu3ZBCIFLly6huLgYRqNR7ViqO3DgALp166Z2DMV47r0llaxYsQJmsxnLly/H8uXLAQAfffSRR7/g6L777sOMGTPw2GOPoby8HDNnzoTBYFA7FjmRYcOGYcaMGRg5ciQkScLChQs9+tY4APTq1QsHDhzAsGHDIIRAQkKCR/+w+6dTp04hIiJC7RiK4ad8ERERuQjeHiciInIRLG0iIiIXwdImIiJyESxtIiIiF8HSJiIichEsbSI3d+7cOfTu3bvSesuWLVVIQ0S3g6VNRETkIljaRB7MbrdjwYIFeOCBB/Dggw863us8NTUVo0ePdnzd9OnTsXHjRpw7dw79+/fHyJEj8dRTT+Ho0aN49NFHMXToUIwcORKnT59W6Tsh8gye/ZZCRB7i8uXLGDRoUKX1devW4cKFC/j6669RVlaG0aNHo0WLFvDx8bnuuU6dOoWPP/4YERERmDFjBp566ikMGDAAmzZtwqFDh9CoUSMFvxMiz8bSJvIA4eHh+Oqrr2RrLVu2RGpqKoYMGQKtVgsfHx8MHDgQ+/btu+Zz4H8KDQ11vE1kTEwMXnvtNezatQu9e/dGr169FP0+iDwdb48TeTC73S47FkLAZrNBkiRc/Q7HVqvV8eer3ye/f//+2LRpE9q1a4dVq1Zh7ty5yocm8mAsbSIP1qVLF2zevBk2mw3FxcX417/+hc6dOyM4OBhnz55FaWkp8vLy8OOPP17z8ZMmTcLhw4cxYsQIvPTSSzhy5EgtfwdEnoW3x4k82PDhw3H69GkMGjQIVqsVAwcORFxcHICKW98PPPAAGjRogA4dOlzz8ePHj8esWbPw/vvvQ6/XY968ebWYnsjz8FO+iIiIXARvjxMREbkIljYREZGLYGkTERG5CJY2ERGRi2BpExERuQiWNhERkYtgaRMREbkIljYREZGL+P+pqy61HCYLMwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plotting line on test data\n",
    "plt.plot(X_test,Y_pred,color='orange',label=\"Prediction\")\n",
    "plt.scatter(X_test,Y_test,color='black',marker='*')\n",
    "plt.xlabel(\"Hours\")\n",
    "plt.ylabel(\"Scores\")\n",
    "plt.title(\"Prediction on test data\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Comparing Actual vs Predicted Scores.¶"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Result</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>20</td>\n",
       "      <td>16.884145</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>27</td>\n",
       "      <td>33.732261</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>69</td>\n",
       "      <td>75.357018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>30</td>\n",
       "      <td>26.794801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>62</td>\n",
       "      <td>60.491033</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Actual     Result\n",
       "0      20  16.884145\n",
       "1      27  33.732261\n",
       "2      69  75.357018\n",
       "3      30  26.794801\n",
       "4      62  60.491033"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_test1 = list(Y_test)\n",
    "prediction=list(Y_pred)\n",
    "df_compare = pd.DataFrame({ 'Actual':Y_test1,'Result':prediction})\n",
    "df_compare"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ACCURACY OF THE MODEL¶"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9454906892105355"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "metrics.r2_score(Y_test,Y_pred)##Goodness of fit Test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Above 94% percentage indicates that above fitted Model is a GOOD MODEL. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Predicting the Error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_squared_error,mean_absolute_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error      =  21.598769307217413\n",
      "Root Mean Squared Error =  4.647447612100368\n",
      "Mean Absolute Error     =  4.647447612100368\n"
     ]
    }
   ],
   "source": [
    "MSE = metrics.mean_squared_error(Y_test,Y_pred)\n",
    "root_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))\n",
    "Abs_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))\n",
    "print(\"Mean Squared Error      = \",MSE)\n",
    "print(\"Root Mean Squared Error = \",root_E)\n",
    "print(\"Mean Absolute Error     = \",Abs_E)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Predicting the score¶"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "predicted score for a student studying 9.25 hours : [93.69173249]\n"
     ]
    }
   ],
   "source": [
    "Prediction_score = linreg.predict([[9.25]])\n",
    "print(\"predicted score for a student studying 9.25 hours :\",Prediction_score)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## CONCLUSION: "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### From the above result we can say that if a studied for 9.25 then student will secured 93.69 MARKS."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
