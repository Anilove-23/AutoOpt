"""Fix known issues in generate_programs.py"""
from pathlib import Path

p = Path(r'c:\Users\ASUS\Desktop\compiler\scripts\generate_programs.py')
src = p.read_text(encoding='utf-8')

fixes = 0

# Fix 1: Python integer division //2 leaking into C++ string
# The linked_list "search" variant uses Python f-string floor division
old1 = "f\"    int target={n}//2; Node* p=head; long long res=0; while(p){{if(p->val==target)res=p->val;p=p->next;}}\""
new1 = "f\"    int target=({n}/2); Node* p=head; long long res=0; while(p){{if(p->val==target)res=p->val;p=p->next;}}\""
if old1 in src:
    src = src.replace(old1, new1)
    fixes += 1
    print("Fixed: floor division //2 in linked_list")
else:
    print("WARNING: floor division pattern not found - may already be fixed or different")

# Fix 2: Dijkstra structured bindings - GCC 6.3 doesn't support auto [d,u] = ...
# Replace with explicit .first/.second access
old2 = '''        auto [d,u]=pq.top(); pq.pop();
        if(d>dist[u]) continue;
        for(auto [w,nb]:dadj[u]) if(dist[u]+w<dist[nb]) {{'''
new2 = '''        long long d=pq.top().first; int u=pq.top().second; pq.pop();
        if(d>dist[u]) continue;
        for(int _i=0;_i<(int)dadj[u].size();_i++) {{ int w=dadj[u][_i].first,nb=dadj[u][_i].second; if(dist[u]+w<dist[nb]) {{'''

if old2 in src:
    src = src.replace(old2, new2)
    # Also need to close the extra brace we added
    old2b = '            dist[nb]=dist[u]+w; pq.push({{dist[nb],nb}});\n        }}\n    }}'
    new2b = '            dist[nb]=dist[u]+w; pq.push({{dist[nb],nb}}); }}\n        }}\n    }}'
    src = src.replace(old2b, new2b)
    fixes += 1
    print("Fixed: Dijkstra structured bindings")
else:
    print("WARNING: Dijkstra pattern not found")

# Fix 3: vector_ops - 'c' declared but not always used (add to wrap_main properly)
# Make the 'add' variant not use a separate 'c' variable to avoid conflicts
old3 = '    if op == "add":\n        kernel = f"    for(int i=0;i<N;i++) c[i]=a[i]+b[i];"\n        result = "c[0]"'
new3 = '    if op == "add":\n        kernel = f"    for(int i=0;i<N;i++) c[i]=a[i]+b[i]; (void)c[0];"\n        result = "c[0]"'
if old3 in src:
    src = src.replace(old3, new3)
    fixes += 1
    print("Fixed: vector_ops unused variable")

# Fix 4: linked_list - the "search" variant declares 'long long res' inside op_code
# but wrap_main will also generate 'long long _result = (long long)(res)'
# This is fine IF op_code is in the algo block (before _result declaration)
# Actually the issue is that op_code ends with 'long long res=...' then wrap_main adds
# 'long long _result = (long long)(res)' right after -> this should work
# The actual issue is: op_code is passed as 'algo' to wrap_main which puts it BEFORE _result
# BUT the declaration of 'res' inside op_code makes it local to that block... wait
# Actually looking at the generated file, op_code IS the algo, and then _result = res
# The problem in the failed file is that op_code contains 'long long res' declared inline
# which is fine since it's in the same scope. The real issue was //2 (fix 1)
# Let's just make the search variant simpler
old4 = '    else:\n        op_code = f"    int target=({n}/2); Node* p=head; long long res=0; while(p){{if(p->val==target)res=p->val;p=p->next;}}"'
new4 = '    else:\n        half = n // 2\n        op_code = f"    int target={half}; Node* p=head; long long res=0; while(p){{if(p->val==target)res=p->val;p=p->next;}}"'
if old4 in src:
    src = src.replace(old4, new4)
    fixes += 1
    print("Fixed: linked_list search variant using Python division safely")
else:
    print("INFO: Fix 4 already applied or pattern different")

p.write_text(src, encoding='utf-8')
print(f"\nTotal fixes applied: {fixes}")
print("Done.")
