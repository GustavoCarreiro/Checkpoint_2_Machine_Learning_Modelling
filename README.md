# 📊 Otimização de KNN Regressor com Dataset Insurance (CP2 - Machine Learning)

Repositório estruturado para documentar o desenvolvimento, engenharia de atributos, otimização de hiperparâmetros via GridSearch e avaliação detalhada de um modelo de **KNN Regressor** aplicado à previsão de custos de seguro médico (*charges*) utilizando o dataset **Insurance**.

---

## 👥 Integrantes do Grupo

* **Rafael Medici** | RM: 576249
* **Gustavo Carreiro** | RM: 576379
* **Ludson Rodrigo Marciano Fernandes** | RM: 574997
* **Lucas Longo Rosette** | RM: 576306
* **Kayky Cayres Vicente da Silva** | RM: 575091

---

## 📂 Visão Geral do Projeto

O objetivo deste projeto foi ir além de um modelo básico de Machine Learning, aplicando técnicas rigorosas de pré-processamento, engenharia de features e Validação Cruzada para maximizar a Acurácia de um algoritmo baseado em distâncias (**K-Nearest Neighbors**).

---

## 🛠️ Roadmap de Desenvolvimento (As 5 Etapas)

### 📍 Etapa 1: Exploração e Qualidade dos Dados
- **Dataset:** 1.338 instâncias e 7 colunas (4 numéricas: `age`, `bmi`, `children`, `charges`; 3 categóricas: `sex`, `smoker`, `region`).
- **Tratamento:** Identificação de ausência de nulos e remoção de uma única linha duplicada encontrada (índices 195 e 581) para garantir a pureza do conjunto de dados.
- **Distribuição do Target (`charges`):** Assimetria à direita (*right-skewed*), onde a média (~13.279) supera a mediana (~9.386) devido a custos elevados.
- **Insights Chave:** A variável `smoker` é a mais determinante no custo do seguro (fumantes pagam em média ~US$ 32.000 contra ~US$ 8.400 de não fumantes), especialmente em indivíduos obesos ($\text{IMC} \ge 30$).

### 📍 Etapa 2: Feature Engineering, Scaling e Encoding
Para otimizar o cálculo de distâncias euclidianas/manhattan do KNN, criaram-se novas variáveis explicativas:
* `obeso`: Indicador binário para $\text{IMC} \ge 30$.
* `fumante`: Mapeamento binário da coluna `smoker`.
* `fumante_obeso`: Interação multiplicativa entre obesidade e tabagismo.
* `fumante_imc`: Produto entre a condição de fumante e o valor do IMC.
* `idade2`: Idade ao quadrado ($\text{age}^2$), capturando o efeito não linear do envelhecimento nos custos.

Aplica-se **One-Hot Encoding** nas categóricas (`sex`, `region`) e escalonamento modular nas numéricas via `ColumnTransformer`.

### 📍 Etapa 3: Divisão dos Dados, Pipeline e Otimização (GridSearchCV)
- **Divisão de Dados:** Separação estanque de 80% treino e 20% teste (`train_test_split`, `random_state=42`) para blindagem contra *data leakage*.
- **Busca via GridSearchCV (5-Fold Cross-Validation):**
  - **Escaladores testados:** `StandardScaler`, `MinMaxScaler`, `RobustScaler`.
  - **Vizinhos ($k$):** Testes de 1 a 50 vizinhos.
  - **Ponderações (`weights`):** `uniform` e `distance`.
  - **Métricas:** `euclidean` e `manhattan`.

#### 🏆 Melhores Hiperparâmetros Encontrados:
* **Escalonador:** `StandardScaler`
* **Métrica:** `euclidean`
* **Vizinhos ($k$):** `26`
* **Ponderação:** `uniform`
* **RMSE Médio no Treino (CV):** `4650.25`

### 📍 Etapa 4: Métricas e Comparação com Modelos de Referência
Avaliação do modelo no conjunto de teste em comparação com modelos *baseline* e a Regressão Linear padrão de aula:

| Modelo | MAE | MSE | RMSE | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Regressão Linear (Aula - Original)** | 4181.19 | 33,596,915.85 | 5796.28 | 0.7836 |
| **Regressão Linear (Aula - Nosso Split)** | 4177.05 | 35,478,020.68 | 5956.34 | 0.8069 |
| **KNN Baseline ($k=5$, sem FE)** | 3476.18 | 29,932,536.69 | 5471.06 | 0.8371 |
| **KNN Otimizado (FE + GridSearch)** | **2496.05** | **19,369,384.44** | **4401.07** | **0.8946** |

* **Ganho do KNN Otimizado:** Redução do **MAE em ~40%** e do **RMSE em ~26%** em comparação com a Regressão Linear do split, além de aumentar o $R^2$ de $0.807$ para **$0.895$**.
* **Controle de Overfitting:** Resultados de treino ($\text{RMSE} = 4487.45$) e teste ($\text{RMSE} = 4401.07$) congruentes, garantindo excelente generalização.

### 📍 Etapa 5: Conclusão, Impacto do Modelo e Próximos Passos

- **Síntese dos Resultados:** O objetivo principal do Checkpoint era superar as métricas de referência apresentadas em aula, meta alcançada com sucesso absoluto pelo **KNN Otimizado**, que obteve o melhor desempenho em todas as quatro métricas avaliadas, destacando-se pelo coeficiente de determinação $R^2$ de **0.895** (frente a 0.807 da Regressão Linear e 0.837 do KNN baseline).
- **Análise de Resíduos e Limitações:**
  - O MAE reduzido (2496) em contraste com o RMSE (4401) evidencia que a maioria das previsões é altamente assertiva, sendo o RMSE puxado por *outliers* pontuais.
  - O modelo apresentou subestimações em dois nichos específicos: um grupo de não fumantes com custos atípicos elevados (não explicados pelas colunas do dataset) e fumantes com custos extremos acima de US$ 50.000 (comportamento inerente à média dos $k=26$ vizinhos).
- **Considerações Finais:** A engenharia de atributos focada na interação entre tabagismo e obesidade, combinada com a busca automatizada de hiperparâmetros, provou ser o diferencial crítico para capturar a não-linearidade do problema de seguros.
