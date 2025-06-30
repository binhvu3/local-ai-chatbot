# Smart Chatbot w/ User Log-In
A simple chatbot application that features user authentication, image classification, and a dark mode toggle. It's powered by a local Ollama instance for generating chat responses.


## 🔧 Prerequisites

- [Docker](https://docs.docker.com/get-docker/)  
- [Ollama CLI](https://ollama.com) (installed on host for local testing; inside container we install it automatically)

---
## 🗂️ Instructions

### 1. Build Docker Image
```bash
./build_docker.sh py-app
 ```

### 2. Run Unit Test
```bash
./build_docker.sh py-app
docker run --rm  py-app ./run_tests.sh
```

### 3. Start Application
```bash
./build_docker.sh py-app
docker run --rm --platform=linux/amd64 -p 5001:5001 py-app
```
Open `http://localhost:5001`

### 4. Create user/password and Log In
- Register username and password
- Loging with the same credentials