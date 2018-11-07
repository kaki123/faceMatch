import json
"""
def remove_linebreak(content):
    n = 0
    while n <= len(content):
        if content[n] == '\n':
            content = content[:n] + content[n+1:]
        n += 1
    return content
"""

def main():
    with open('../testInput/mzheng.json') as f:
        data = json.load(f)
        print(data)

if __name__ == "__main__":
    main()