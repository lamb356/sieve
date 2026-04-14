pub const ALIASES: &[(&str, &str)] = &[
    ("std", "standard library"),
    ("lib", "library"),
    ("pkg", "package"),
    ("mod", "module"),
    ("fn", "function"),
    ("iface", "interface"),
    ("impl", "implementation"),
    ("ctx", "context"),
    ("arg", "argument"),
    ("param", "parameter"),
    ("api", "application programming interface"),
    ("cli", "command line interface"),
    ("kwarg", "keyword argument"),
    ("opts", "options"),
    ("util", "utility"),
    ("iter", "iterator"),
    ("msg", "message"),
    ("evt", "event"),
    ("svc", "service"),
    ("repo", "repository"),
    ("config", "configuration"),
    ("err", "error"),
    ("exc", "exception"),
    ("auth", "authentication"),
    ("authn", "authentication"),
    ("authz", "authorization"),
    ("perm", "permission"),
    ("cred", "credential"),
    ("ser", "serialization"),
    ("deser", "deserialization"),
    ("async", "asynchronous"),
    ("sync", "synchronous"),
    ("coro", "coroutine"),
    ("chan", "channel"),
    ("sem", "semaphore"),
    ("mutex", "mutual exclusion"),
    ("rwlock", "read write lock"),
    ("condvar", "condition variable"),
    ("fut", "future"),
    ("prom", "promise"),
    ("wg", "wait group"),
    ("mpsc", "multi producer single consumer"),
    ("req", "request"),
    ("resp", "response"),
    ("hdr", "header"),
    ("addr", "address"),
    ("url", "uniform resource locator"),
    ("tls", "transport layer security"),
    ("cert", "certificate"),
    ("conn", "connection"),
    ("sock", "socket"),
    ("rpc", "remote procedure call"),
    ("grpc", "google remote procedure call"),
    ("db", "database"),
    ("sql", "structured query language"),
    ("orm", "object relational mapping"),
    ("txn", "transaction"),
    ("mig", "migration"),
    ("idx", "index"),
    ("pk", "primary key"),
    ("fk", "foreign key"),
    ("col", "column"),
    ("uuid", "universally unique identifier"),
    ("kv", "key value"),
    ("ttl", "time to live"),
    ("cfg", "configuration"),
    ("env", "environment"),
    ("envvar", "environment variable"),
    ("venv", "virtual environment"),
    ("spec", "specification"),
    ("ut", "unit test"),
    ("integ", "integration test"),
    ("e2e", "end to end"),
    ("ci", "continuous integration"),
    ("cd", "continuous deployment"),
    ("iac", "infrastructure as code"),
    ("k8s", "kubernetes"),
    ("infra", "infrastructure"),
    ("iam", "identity access management"),
    ("rbac", "role based access control"),
    ("fs", "filesystem"),
    ("io", "input output"),
    ("fd", "file descriptor"),
    ("dir", "directory"),
    ("tmp", "temporary"),
    ("buf", "buffer"),
    ("vec", "vector"),
    ("dict", "dictionary"),
    ("str", "string"),
    ("bool", "boolean"),
    ("arr", "array"),
    ("obj", "object"),
    ("elem", "element"),
    ("val", "value"),
    ("num", "number"),
    ("int", "integer"),
    ("ref", "reference"),
    ("ptr", "pointer"),
    ("opt", "option"),
    ("res", "result"),
];

use std::collections::HashMap;

pub struct AliasLexicon {
    pub map: HashMap<String, Vec<String>>,
}

impl AliasLexicon {
    pub fn built_in() -> Self {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for &(short, long) in ALIASES {
            map.entry(short.to_string())
                .or_default()
                .push(long.to_string());
            map.entry(long.to_string())
                .or_default()
                .push(short.to_string());
        }
        Self { map }
    }

    pub fn lookup(&self, term: &str) -> Option<&[String]> {
        self.map.get(term).map(|v| v.as_slice())
    }

    pub fn same_alias_family(&self, a: &str, b: &str) -> bool {
        if let Some(aliases) = self.map.get(a) {
            aliases.iter().any(|x| x == b)
        } else {
            false
        }
    }
}
