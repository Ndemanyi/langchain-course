# Groq Model Switching — simulation LangChain

## Présentation

Cet exercice reproduit le fonctionnement général de l’intégration `langchain-groq` sans appeler réellement l’API Groq. Des objets simulés (*mocks*) permettent de pratiquer la configuration d’une clé API, la création de modèles, l’envoi d’un prompt et la comparaison de réponses.

> **Important :** aucune requête réseau n’est effectuée et aucune réponse n’est générée par un véritable LLM. Les réponses sont prédéfinies dans la méthode `ChatGroq.invoke()`.

## Objectifs pédagogiques

- Manipuler une variable d’environnement avec `os.environ`.
- Instancier une classe avec différentes configurations.
- Utiliser une méthode commune pour interroger plusieurs modèles.
- Structurer un message LangChain avec une liste de tuples.
- Composer plusieurs fonctions dans une fonction d’orchestration.
- Regrouper des résultats dans un dictionnaire.
- Comprendre la propagation et le traitement des exceptions.

## Fonctionnement général

Le programme suit les étapes suivantes :

1. place une fausse clé de test dans `GROQ_API_KEY`;
2. vérifie que cette variable d’environnement existe;
3. crée une instance simulée de Llama 4;
4. crée une instance simulée de Llama 3.3;
5. envoie le même prompt aux deux instances;
6. rassemble leurs réponses dans un dictionnaire;
7. affiche les résultats ou un message d’aide en cas d’exception.

## Structure du dossier

```text
groq_model_switching/
├── exercise.py
└── README.md
```

## Classes

### `ChatGroq`

Classe simulée qui imite une petite partie du comportement de la véritable classe `ChatGroq` fournie par `langchain-groq`.

#### Constructeur `__init__(model, temperature=0, max_retries=2)`

Le constructeur initialise :

- `model` : nom exact du modèle à simuler;
- `temperature` : niveau de variation de la réponse;
- `max_retries` : nombre maximal de nouvelles tentatives;
- `valid_models` : liste des noms acceptés dans l’exercice.

Il lève une `ValueError` lorsque le nom reçu ne se trouve pas dans `valid_models`.

#### Méthode `invoke(messages)`

Cette méthode :

1. vérifie que `messages` est une liste non vide;
2. choisit une réponse prédéfinie selon le modèle et la température;
3. retourne la réponse dans un objet `MockAIMessage`.

Dans cet exercice, le contenu réel du prompt n’est pas analysé.

### `MockAIMessage`

Classe simulant le message retourné par un modèle conversationnel. Son attribut `content` contient le texte de la réponse.

## Fonctions

### `implement_set_api_key(api_key)`

Enregistre la valeur reçue dans la variable d’environnement `GROQ_API_KEY` :

```python
os.environ["GROQ_API_KEY"] = api_key
```

La fonction modifie l’environnement du processus Python et n’a pas besoin de retourner la clé.

### `check_api_key()`

Vérifie que `GROQ_API_KEY` existe dans `os.environ`. Elle lève une exception si la variable est absente.

### `implement_llama_4_model()`

Crée et retourne une instance de `ChatGroq` configurée avec :

- modèle : `llama-4-8b-instant`;
- température : `0`, pour une réponse stable dans la simulation.

### `implement_llama_3_3_model()`

Crée et retourne une instance de `ChatGroq` configurée avec :

- modèle : `llama-3.3-70b-versatile`;
- température : `0.5`, afin de déclencher la réponse créative simulée.

### `implement_query_model(model, prompt)`

Transforme le prompt en message de conversation :

```python
messages = [("human", prompt)]
```

La fonction appelle ensuite `model.invoke(messages)` et retourne uniquement `response.content`.

### `implement_compare_models(prompt)`

Fonction d’orchestration qui :

1. crée les deux modèles;
2. leur envoie le même prompt;
3. place les deux réponses dans un dictionnaire;
4. retourne ce dictionnaire.

Cette fonction rend les réponses disponibles côte à côte, mais elle ne calcule pas automatiquement laquelle est la meilleure.

### `main()`

Point d’entrée du programme. Elle exécute les différentes étapes de démonstration et affiche les résultats.

Le bloc `try/except` intercepte la première exception rencontrée. Il affiche une indication particulière lorsque le message concerne `GROQ_API_KEY`, sinon il affiche un conseil général sur les fonctions et les noms de modèles.

## Flux des données

```text
prompt
  → [("human", prompt)]
  → ChatGroq.invoke(messages)
  → MockAIMessage
  → response.content
  → dictionnaire de comparaison
```

## Exécution

À partir de la racine du dépôt, avec l’environnement virtuel activé :

```powershell
python exercises/groq_model_switching/exercise.py
```

Avec `uv`, il est également possible d’utiliser :

```powershell
uv run python exercises/groq_model_switching/exercise.py
```

## Résultat attendu

Le terminal doit confirmer :

- la présence de la clé API simulée;
- le fonctionnement de la configuration Llama 4;
- le fonctionnement de la configuration Llama 3.3;
- l’affichage des deux réponses dans le dictionnaire de comparaison;
- la réussite de toutes les implémentations.

## Sécurité

La valeur `mock_api_key_for_testing` est volontairement fictive. Dans un projet réel :

- ne jamais écrire une véritable clé API dans le code;
- conserver les secrets dans un fichier `.env` ou dans un gestionnaire de secrets;
- ajouter `.env` au fichier `.gitignore`;
- vérifier les fichiers préparés avec `git status` avant chaque commit.

## Limites de la simulation

- Aucun appel réel à Groq n’est effectué.
- Les réponses sont codées en dur.
- Le prompt est placé dans la structure de messages, mais son contenu n’influence pas la réponse simulée.
- La validité réelle des modèles et de la clé API n’est pas vérifiée auprès de Groq.

## Prochaine étape

Après avoir maîtrisé cette simulation, la prochaine étape consiste à remplacer les classes mock par la véritable intégration `langchain-groq`, à charger la clé depuis un environnement sécurisé et à gérer les erreurs réelles d’authentification, de réseau et de limite de requêtes.
