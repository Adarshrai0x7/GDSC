{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "6jdvrVohccmr"
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import linear_model\n",
    "from surprise.model_selection import train_test_split\n",
    "from surprise import Reader, Dataset, SVD, accuracy\n",
    "from sklearn.metrics import precision_recall_fscore_support\n",
    "from sklearn.preprocessing import MultiLabelBinarizer\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "Lbc9ib3YbnP-"
   },
   "outputs": [],
   "source": [
    "movies_data =  pd.read_csv('movies.csv')\n",
    "ratings_data = pd.read_csv('ratings.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 9742 entries, 0 to 9741\n",
      "Data columns (total 3 columns):\n",
      " #   Column   Non-Null Count  Dtype \n",
      "---  ------   --------------  ----- \n",
      " 0   movieId  9742 non-None   int64 \n",
      " 1   title    9742 non-None   object\n",
      " 2   genres   9742 non-None   object\n",
      "dtypes: int64(1), object(2)\n",
      "memory usage: 228.5+ KB\n"
     ]
    }
   ],
   "source": [
    "movies_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "393zeQRBe5N1",
    "outputId": "bfa18219-b827-4068-9df4-2e848afcd5e2"
   },
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
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Jumanji (1995)</td>\n",
       "      <td>Adventure|Children|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Grumpier Old Men (1995)</td>\n",
       "      <td>Comedy|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Waiting to Exhale (1995)</td>\n",
       "      <td>Comedy|Drama|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Father of the Bride Part II (1995)</td>\n",
       "      <td>Comedy</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   movieId                               title  \\\n",
       "0        1                    Toy Story (1995)   \n",
       "1        2                      Jumanji (1995)   \n",
       "2        3             Grumpier Old Men (1995)   \n",
       "3        4            Waiting to Exhale (1995)   \n",
       "4        5  Father of the Bride Part II (1995)   \n",
       "\n",
       "                                        genres  \n",
       "0  Adventure|Animation|Children|Comedy|Fantasy  \n",
       "1                   Adventure|Children|Fantasy  \n",
       "2                               Comedy|Romance  \n",
       "3                         Comedy|Drama|Romance  \n",
       "4                                       Comedy  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "id": "iQ4kyucOcQ_s",
    "outputId": "80670340-23c3-4faf-e6b5-7f3fbaf3f99e"
   },
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
       "      <th>userId</th>\n",
       "      <th>movieId</th>\n",
       "      <th>rating</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>964982703</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>4.0</td>\n",
       "      <td>964981247</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>4.0</td>\n",
       "      <td>964982224</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>47</td>\n",
       "      <td>5.0</td>\n",
       "      <td>964983815</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>50</td>\n",
       "      <td>5.0</td>\n",
       "      <td>964982931</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   userId  movieId  rating  timestamp\n",
       "0       1        1     4.0  964982703\n",
       "1       1        3     4.0  964981247\n",
       "2       1        6     4.0  964982224\n",
       "3       1       47     5.0  964983815\n",
       "4       1       50     5.0  964982931"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "ratings_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 241
    },
    "id": "ddz_1pqTjmNn",
    "outputId": "4a3b0387-3ca9-4ee7-928d-6991710fb9ba"
   },
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
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Jumanji (1995)</td>\n",
       "      <td>Adventure|Children|Fantasy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Grumpier Old Men (1995)</td>\n",
       "      <td>Comedy|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Waiting to Exhale (1995)</td>\n",
       "      <td>Comedy|Drama|Romance</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Father of the Bride Part II (1995)</td>\n",
       "      <td>Comedy</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   movieId                               title  \\\n",
       "0        1                    Toy Story (1995)   \n",
       "1        2                      Jumanji (1995)   \n",
       "2        3             Grumpier Old Men (1995)   \n",
       "3        4            Waiting to Exhale (1995)   \n",
       "4        5  Father of the Bride Part II (1995)   \n",
       "\n",
       "                                        genres  \n",
       "0  Adventure|Animation|Children|Comedy|Fantasy  \n",
       "1                   Adventure|Children|Fantasy  \n",
       "2                               Comedy|Romance  \n",
       "3                         Comedy|Drama|Romance  \n",
       "4                                       Comedy  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 209
    },
    "id": "4fm-gd6pj-JM",
    "outputId": "6cbd0e5e-85d8-4eb6-d305-ef33afa9bd4a"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "movieId    0\n",
       "title      0\n",
       "genres     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_data.isNone().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "2eAu0IqQkSGA"
   },
   "outputs": [],
   "source": [
    "movies_data.dropna(inplace = True ,axis = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 209
    },
    "id": "8W8rHV9Dk7dU",
    "outputId": "39dc95a0-2af3-4b51-bdc7-78da6ba00a30"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "movieId    0\n",
       "title      0\n",
       "genres     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_data.isNone().sum()"
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
       "userId       0\n",
       "movieId      0\n",
       "rating       0\n",
       "timestamp    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "ratings_data.isNone().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_data2= movies_data.merge(ratings_data, on='movieId', how='inner')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(100836, 6)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_data2.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "upebrjepryf_"
   },
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
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "      <th>userId</th>\n",
       "      <th>rating</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "      <td>1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>964982703</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "      <td>5</td>\n",
       "      <td>4.0</td>\n",
       "      <td>847434962</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "      <td>7</td>\n",
       "      <td>4.5</td>\n",
       "      <td>1106635946</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "      <td>15</td>\n",
       "      <td>2.5</td>\n",
       "      <td>1510577970</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n",
       "      <td>17</td>\n",
       "      <td>4.5</td>\n",
       "      <td>1305696483</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   movieId             title                                       genres  \\\n",
       "0        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   \n",
       "1        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   \n",
       "2        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   \n",
       "3        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   \n",
       "4        1  Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   \n",
       "\n",
       "   userId  rating   timestamp  \n",
       "0       1     4.0   964982703  \n",
       "1       5     4.0   847434962  \n",
       "2       7     4.5  1106635946  \n",
       "3      15     2.5  1510577970  \n",
       "4      17     4.5  1305696483  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_data2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "n9-_FZPPlA-S"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "title\n",
       "Gena the Crocodile (1969)              5.0\n",
       "True Stories (1986)                    5.0\n",
       "Cosmic Scrat-tastrophe (2015)          5.0\n",
       "Love and Pigeons (1985)                5.0\n",
       "Red Sorghum (Hong gao liang) (1987)    5.0\n",
       "Name: rating, dtype: float64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_data2.groupby('title')['rating'].mean().sort_values(ascending=False).head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "title\n",
       "Forrest Gump (1994)                 329\n",
       "Shawshank Redemption, The (1994)    317\n",
       "Pulp Fiction (1994)                 307\n",
       "Silence of the Lambs, The (1991)    279\n",
       "Matrix, The (1999)                  278\n",
       "Name: rating, dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_data2.groupby('title')['rating'].count().sort_values(ascending=False).head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
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
       "      <th>rating</th>\n",
       "      <th>no of ratings</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>'71 (2014)</th>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Hellboy': The Seeds of Creation (2004)</th>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Round Midnight (1986)</th>\n",
       "      <td>3.5</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Salem's Lot (2004)</th>\n",
       "      <td>5.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Til There Was You (1997)</th>\n",
       "      <td>4.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                         rating  no of ratings\n",
       "title                                                         \n",
       "'71 (2014)                                  4.0              1\n",
       "'Hellboy': The Seeds of Creation (2004)     4.0              1\n",
       "'Round Midnight (1986)                      3.5              2\n",
       "'Salem's Lot (2004)                         5.0              1\n",
       "'Til There Was You (1997)                   4.0              2"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ratings = pd.DataFrame(movies_data2.groupby('title')['rating'].mean())\n",
    "ratings['no of ratings'] = pd.DataFrame(movies_data2.groupby('title')['rating'].count())\n",
    "ratings.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
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
       "      <th>rating</th>\n",
       "      <th>no of ratings</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Forrest Gump (1994)</th>\n",
       "      <td>4.164134</td>\n",
       "      <td>329</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Shawshank Redemption, The (1994)</th>\n",
       "      <td>4.429022</td>\n",
       "      <td>317</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pulp Fiction (1994)</th>\n",
       "      <td>4.197068</td>\n",
       "      <td>307</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Silence of the Lambs, The (1991)</th>\n",
       "      <td>4.161290</td>\n",
       "      <td>279</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Matrix, The (1999)</th>\n",
       "      <td>4.192446</td>\n",
       "      <td>278</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                    rating  no of ratings\n",
       "title                                                    \n",
       "Forrest Gump (1994)               4.164134            329\n",
       "Shawshank Redemption, The (1994)  4.429022            317\n",
       "Pulp Fiction (1994)               4.197068            307\n",
       "Silence of the Lambs, The (1991)  4.161290            279\n",
       "Matrix, The (1999)                4.192446            278"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# recommending similar movies\n",
    "ratings.sort_values('no of ratings' , ascending=False).head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "movies_feature =movies_data2.pivot_table(index='title',columns='userId',values='rating')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
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
       "      <th>userId</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "      <th>...</th>\n",
       "      <th>601</th>\n",
       "      <th>602</th>\n",
       "      <th>603</th>\n",
       "      <th>604</th>\n",
       "      <th>605</th>\n",
       "      <th>606</th>\n",
       "      <th>607</th>\n",
       "      <th>608</th>\n",
       "      <th>609</th>\n",
       "      <th>610</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>'71 (2014)</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Hellboy': The Seeds of Creation (2004)</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Round Midnight (1986)</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Salem's Lot (2004)</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Til There Was You (1997)</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 610 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "userId                                   1    2    3    4    5    6    7    \\\n",
       "title                                                                        \n",
       "'71 (2014)                               NaN  NaN  NaN  NaN  NaN  NaN  NaN   \n",
       "'Hellboy': The Seeds of Creation (2004)  NaN  NaN  NaN  NaN  NaN  NaN  NaN   \n",
       "'Round Midnight (1986)                   NaN  NaN  NaN  NaN  NaN  NaN  NaN   \n",
       "'Salem's Lot (2004)                      NaN  NaN  NaN  NaN  NaN  NaN  NaN   \n",
       "'Til There Was You (1997)                NaN  NaN  NaN  NaN  NaN  NaN  NaN   \n",
       "\n",
       "userId                                   8    9    10   ...  601  602  603  \\\n",
       "title                                                   ...                  \n",
       "'71 (2014)                               NaN  NaN  NaN  ...  NaN  NaN  NaN   \n",
       "'Hellboy': The Seeds of Creation (2004)  NaN  NaN  NaN  ...  NaN  NaN  NaN   \n",
       "'Round Midnight (1986)                   NaN  NaN  NaN  ...  NaN  NaN  NaN   \n",
       "'Salem's Lot (2004)                      NaN  NaN  NaN  ...  NaN  NaN  NaN   \n",
       "'Til There Was You (1997)                NaN  NaN  NaN  ...  NaN  NaN  NaN   \n",
       "\n",
       "userId                                   604  605  606  607  608  609  610  \n",
       "title                                                                       \n",
       "'71 (2014)                               NaN  NaN  NaN  NaN  NaN  NaN  4.0  \n",
       "'Hellboy': The Seeds of Creation (2004)  NaN  NaN  NaN  NaN  NaN  NaN  NaN  \n",
       "'Round Midnight (1986)                   NaN  NaN  NaN  NaN  NaN  NaN  NaN  \n",
       "'Salem's Lot (2004)                      NaN  NaN  NaN  NaN  NaN  NaN  NaN  \n",
       "'Til There Was You (1997)                NaN  NaN  NaN  NaN  NaN  NaN  NaN  \n",
       "\n",
       "[5 rows x 610 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_feature.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_feature.fillna(0,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
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
       "      <th>userId</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "      <th>...</th>\n",
       "      <th>601</th>\n",
       "      <th>602</th>\n",
       "      <th>603</th>\n",
       "      <th>604</th>\n",
       "      <th>605</th>\n",
       "      <th>606</th>\n",
       "      <th>607</th>\n",
       "      <th>608</th>\n",
       "      <th>609</th>\n",
       "      <th>610</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>'71 (2014)</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Hellboy': The Seeds of Creation (2004)</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Round Midnight (1986)</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Salem's Lot (2004)</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>'Til There Was You (1997)</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 610 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "userId                                   1    2    3    4    5    6    7    \\\n",
       "title                                                                        \n",
       "'71 (2014)                               0.0  0.0  0.0  0.0  0.0  0.0  0.0   \n",
       "'Hellboy': The Seeds of Creation (2004)  0.0  0.0  0.0  0.0  0.0  0.0  0.0   \n",
       "'Round Midnight (1986)                   0.0  0.0  0.0  0.0  0.0  0.0  0.0   \n",
       "'Salem's Lot (2004)                      0.0  0.0  0.0  0.0  0.0  0.0  0.0   \n",
       "'Til There Was You (1997)                0.0  0.0  0.0  0.0  0.0  0.0  0.0   \n",
       "\n",
       "userId                                   8    9    10   ...  601  602  603  \\\n",
       "title                                                   ...                  \n",
       "'71 (2014)                               0.0  0.0  0.0  ...  0.0  0.0  0.0   \n",
       "'Hellboy': The Seeds of Creation (2004)  0.0  0.0  0.0  ...  0.0  0.0  0.0   \n",
       "'Round Midnight (1986)                   0.0  0.0  0.0  ...  0.0  0.0  0.0   \n",
       "'Salem's Lot (2004)                      0.0  0.0  0.0  ...  0.0  0.0  0.0   \n",
       "'Til There Was You (1997)                0.0  0.0  0.0  ...  0.0  0.0  0.0   \n",
       "\n",
       "userId                                   604  605  606  607  608  609  610  \n",
       "title                                                                       \n",
       "'71 (2014)                               0.0  0.0  0.0  0.0  0.0  0.0  4.0  \n",
       "'Hellboy': The Seeds of Creation (2004)  0.0  0.0  0.0  0.0  0.0  0.0  0.0  \n",
       "'Round Midnight (1986)                   0.0  0.0  0.0  0.0  0.0  0.0  0.0  \n",
       "'Salem's Lot (2004)                      0.0  0.0  0.0  0.0  0.0  0.0  0.0  \n",
       "'Til There Was You (1997)                0.0  0.0  0.0  0.0  0.0  0.0  0.0  \n",
       "\n",
       "[5 rows x 610 columns]"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_feature.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9719, 610)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from scipy.sparse import csr_matrix\n",
    "features_matrix = csr_matrix(movies_feature.values)\n",
    "features_matrix.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "from surprise import SVD, Dataset, Reader\n",
    "from surprise.model_selection import cross_validate\n",
    "reader = Reader(rating_scale=(0.5, 5.0))\n",
    "data = Dataset.load_from_df(ratings_data[[\"userId\", \"movieId\", \"rating\"]], reader)\n",
    "trainset, testset = train_test_split(data, test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<surprise.prediction_algorithms.matrix_factorization.SVD at 0x1e89b95b430>"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "svd_model = SVD()\n",
    "svd_model.fit(trainset)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4.343773105694895\n"
     ]
    }
   ],
   "source": [
    "def predict_rating(user_id, movie_id):\n",
    "    return svd_model.predict(user_id, movie_id).est\n",
    "\n",
    "print(predict_rating(1, 2))  # Predict rating for User 1 and Movie 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
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
       "      <th>movieId</th>\n",
       "      <th>title</th>\n",
       "      <th>genres</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>[Adventure, Animation, Children, Comedy, Fantasy]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   movieId             title  \\\n",
       "0        1  Toy Story (1995)   \n",
       "\n",
       "                                              genres  \n",
       "0  [Adventure, Animation, Children, Comedy, Fantasy]  "
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_data[\"genres\"] = movies_data[\"genres\"].apply(lambda x: x.split(\"|\"))\n",
    "movies_data.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Movies similar to 'Toy Story (1995)':\n",
      "Monsters, Inc. (2001)\n",
      "Moana (2016)\n",
      "Asterix and the Vikings (Astérix et les Vikings) (2006)\n",
      "Turbo (2013)\n",
      "The Good Dinosaur (2015)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['Monsters, Inc. (2001)',\n",
       " 'Moana (2016)',\n",
       " 'Asterix and the Vikings (Astérix et les Vikings) (2006)',\n",
       " 'Turbo (2013)',\n",
       " 'The Good Dinosaur (2015)']"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.neighbors import NearestNeighbors\n",
    "from sklearn.preprocessing import MultiLabelBinarizer\n",
    "\n",
    "\n",
    "\n",
    "# Apply MultiLabelBinarizer to encode genres as binary features\n",
    "mlb = MultiLabelBinarizer()\n",
    "genre_features = mlb.fit_transform(movies_data[\"genres\"])\n",
    "\n",
    "# Fit k-NN on genre features\n",
    "knn_genre = NearestNeighbors(metric=\"cosine\", algorithm=\"brute\")\n",
    "knn_genre.fit(genre_features)\n",
    "\n",
    "# Function to recommend similar movies based on genres\n",
    "def recommend_movies_by_genre(movie_title, n_recommendations=5):\n",
    "    if movie_title not in movies_data[\"title\"].values:\n",
    "        print(\"Movie not found in dataset!\")\n",
    "        return []\n",
    "    \n",
    "    # Get index of the movie\n",
    "    movie_idx = movies_data[movies_data[\"title\"] == movie_title].index[0]\n",
    "    \n",
    "    # Find similar movies\n",
    "    distances, indices = knn_genre.kneighbors([genre_features[movie_idx]], n_neighbors=n_recommendations+1)\n",
    "\n",
    "    # Print recommended movies\n",
    "    print(f\"Movies similar to '{movie_title}':\")\n",
    "    recommended_movies = [movies_data.iloc[idx][\"title\"] for idx in indices.flatten()[1:]]\n",
    "    for movie in recommended_movies:\n",
    "        print(movie)\n",
    "\n",
    "    return recommended_movies\n",
    "\n",
    "# Example usage: Recommend movies similar to \"Toy Story (1995)\"\n",
    "recommend_movies_by_genre(\"Toy Story (1995)\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Hybrid recommendations for User 1 based on 'Toy Story (1995)':\n",
      "Monsters, Inc. (2001)\n",
      "The Good Dinosaur (2015)\n",
      "Moana (2016)\n",
      "Turbo (2013)\n",
      "Asterix and the Vikings (Astérix et les Vikings) (2006)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['Monsters, Inc. (2001)',\n",
       " 'The Good Dinosaur (2015)',\n",
       " 'Moana (2016)',\n",
       " 'Turbo (2013)',\n",
       " 'Asterix and the Vikings (Astérix et les Vikings) (2006)']"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def hybrid_recommendation(user_id, movie_title, n_recommendations=5):\n",
    "    # Step 1: Get SVD predictions for all movies\n",
    "    svd_predictions = []\n",
    "    for movie_id in movies_data['movieId'].unique():\n",
    "        svd_predictions.append((movie_id, svd_model.predict(user_id, movie_id).est))\n",
    "    \n",
    "    # Sort predictions by estimated rating\n",
    "    svd_predictions.sort(key=lambda x: x[1], reverse=True)\n",
    "    \n",
    "    # Step 2: Get k-NN recommendations based on genres\n",
    "    if movie_title not in movies_data[\"title\"].values:\n",
    "        print(\"Movie not found in dataset!\")\n",
    "        return []\n",
    "    \n",
    "    movie_idx = movies_data[movies_data[\"title\"] == movie_title].index[0]\n",
    "    distances, indices = knn_genre.kneighbors([genre_features[movie_idx]], n_neighbors=n_recommendations+1)\n",
    "    knn_recommendations = [movies_data.iloc[idx][\"movieId\"] for idx in indices.flatten()[1:]]\n",
    "    \n",
    "    # Step 3: Combine SVD and k-NN recommendations\n",
    "    hybrid_recommendations = []\n",
    "    for movie_id, rating in svd_predictions:\n",
    "        if movie_id in knn_recommendations:\n",
    "            hybrid_recommendations.append((movie_id, rating))\n",
    "        if len(hybrid_recommendations) >= n_recommendations:\n",
    "            break\n",
    "    \n",
    "    # Get movie titles for the final recommendations\n",
    "    recommended_movies = [movies_data[movies_data[\"movieId\"] == movie_id][\"title\"].values[0] for movie_id, _ in hybrid_recommendations]\n",
    "    \n",
    "    print(f\"Hybrid recommendations for User {user_id} based on '{movie_title}':\")\n",
    "    for movie in recommended_movies:\n",
    "        print(movie)\n",
    "    \n",
    "    return recommended_movies\n",
    "\n",
    "# Example usage: Get hybrid recommendations for User 1 based on \"Toy Story (1995)\"\n",
    "hybrid_recommendation(1, \"Toy Story (1995)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE: 0.8746\n",
      "Root Mean Squared Error (RMSE): 0.8745644100705388\n",
      "Precision: 0.7939\n",
      "Recall: 0.6886\n",
      "F1-Score: 0.7375\n"
     ]
    }
   ],
   "source": [
    "predictions = svd_model.test(testset)\n",
    "\n",
    "# Compute RMSE\n",
    "rmse = accuracy.rmse(predictions)\n",
    "print(\"Root Mean Squared Error (RMSE):\", rmse)\n",
    "\n",
    "# Convert predictions to binary (relevant or not relevant)\n",
    "threshold = 3.5  # Consider ratings >= 3.5 as relevant\n",
    "y_true = [1 if true_r >= threshold else 0 for (_, _, true_r, _, _) in predictions]\n",
    "y_pred = [1 if est >= threshold else 0 for (_, _, _, est, _) in predictions]\n",
    "\n",
    "# Compute Precision, Recall, F1-Score\n",
    "precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')\n",
    "\n",
    "print(f\"Precision: {precision:.4f}\")\n",
    "print(f\"Recall: {recall:.4f}\")\n",
    "print(f\"F1-Score: {f1:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "ename": "OSError",
     "evalue": "[Errno 22] Invalid argument: 'C:\\\\Jupyter\\x07pp.ipynb'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mOSError\u001b[0m                                   Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[33], line 5\u001b[0m\n\u001b[0;32m      2\u001b[0m file_path \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mC:\u001b[39m\u001b[38;5;124m\\\u001b[39m\u001b[38;5;124mJupyter\u001b[39m\u001b[38;5;130;01m\\a\u001b[39;00m\u001b[38;5;124mpp.ipynb\u001b[39m\u001b[38;5;124m'\u001b[39m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;66;03m# Read the content of the file\u001b[39;00m\n\u001b[1;32m----> 5\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mfile_path\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mr\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mas\u001b[39;00m file:\n\u001b[0;32m      6\u001b[0m     content \u001b[38;5;241m=\u001b[39m file\u001b[38;5;241m.\u001b[39mread()\n\u001b[0;32m      8\u001b[0m \u001b[38;5;66;03m# Replace all occurrences of 'None' with 'None'\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\adars\\anaconda3\\envs\\tf\\lib\\site-packages\\IPython\\core\\interactiveshell.py:310\u001b[0m, in \u001b[0;36m_modified_open\u001b[1;34m(file, *args, **kwargs)\u001b[0m\n\u001b[0;32m    303\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m file \u001b[38;5;129;01min\u001b[39;00m {\u001b[38;5;241m0\u001b[39m, \u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m2\u001b[39m}:\n\u001b[0;32m    304\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    305\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mIPython won\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mt let you open fd=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mfile\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m by default \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    306\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mas it is likely to crash IPython. If you know what you are doing, \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    307\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124myou can use builtins\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m open.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    308\u001b[0m     )\n\u001b[1;32m--> 310\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m io_open(file, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "\u001b[1;31mOSError\u001b[0m: [Errno 22] Invalid argument: 'C:\\\\Jupyter\\x07pp.ipynb'"
     ]
    }
   ],
   "source": [
    "# Define the path to your Python file\n",
    "file_path = 'C:\\Jupyter\\app.ipynb'\n",
    "\n",
    "# Read the content of the file\n",
    "with open(file_path, 'r') as file:\n",
    "    content = file.read()\n",
    "\n",
    "# Replace all occurrences of 'None' with 'None'\n",
    "updated_content = content.replace('None', 'None')\n",
    "\n",
    "# Write the updated content back to the file\n",
    "with open(file_path, 'w') as file:\n",
    "    file.write(updated_content)\n",
    "\n",
    "print(f\"Replaced 'None' with 'None' in {file_path}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "tf",
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
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
