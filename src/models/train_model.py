from prefect import flow, task

@task
def greet(name: str):
    print(f"Hello, {name}!")

@flow
def greeting_flow(name: str = "World"):
    greet(name)

# Run it
if __name__ == "__main__":
    greeting_flow(name="Alice")
