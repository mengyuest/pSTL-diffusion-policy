import torch

def clip(x, a, b):
    return max(min(x, b), a)

def softmax(x, tau, d, dim=1):  # assume x (n, t)
    if x.shape[1]==0:
        return torch.ones(x.shape[0], 1).to(x.device) * -float('inf')  # TODO(debug)
    else:
        if d is not None and "hard" in d and d["hard"]:
            return torch.max(x, dim=dim, keepdim=True)[0]
        else:
            return torch.logsumexp(x * tau, dim=dim, keepdim=True) / tau

def softmin(x, tau, d, dim=1):
    if x.shape[1]==0:
        return torch.ones(x.shape[0], 1).to(x.device) * -float('inf')  # TODO(debug)
    else:
        return -softmax(-x, tau, d, dim)

def softmax_pairs(x, y, tau, d): # x (n, t), y (n, t)
    xy = torch.stack([x, y], dim=1)   
    return softmax(xy, tau, d).squeeze(1)

def softmin_pairs(x, y, tau, d):
    return -softmax_pairs(-x, -y, tau, d)

class STLFormula():
    def __init__(self, ts=None, te=None, node=None, lhs=None, rhs=None, lists=None, operator=None):
        self.ts = ts
        self.te = te
        self.node = node
        self.lhs = lhs
        self.rhs = rhs
        self.lists = lists
        self.operator = operator  # {"symbol": "dbg", "word": "DEBUG"}
        self.format = "symbol"   # ["symbol", "word"]

    def __call__(self, x, tau):  # compute the robustness score (based on the upstream up_ts, up_te, and self.ts, self.te)
        raise NotImplementedError
    
    def __str__(self):
        ops = self.operator[self.format]
        if self.ts is not None:
            ops = "%s[%d:%d]"%(ops, self.ts, self.te+1)
        if self.node is not None:
            return "%s (%s)"%(ops, self.node)
        elif self.lhs is not None:
            return "(%s) %s (%s)"%(self.lhs, ops, self.rhs)
        elif self.lists is not None:
            return "%s {%s}"%(ops, ",".join(["|%s|"%x for x in self.lists]))
        else:
            raise NotImplementedError

    def children(self):
        if self.node is not None:
            return [self.node]
        else:
            return [self.lhs, self.rhs]
    
    def update_format(self, format):
        self.format = format
        for child in self.children():
            if hasattr(child, "update_format"):
                child.update_format(format)

    def build(self, s):
        raise NotImplementedError

class AP:
    n_aps = 0
    def __init__(self, expression, comment=None):
        self.expression = expression
        self.comment = comment
        self.apid = AP.n_aps
        AP.n_aps += 1
    
    def __call__(self, x, tau, d=None):  # compute the robustness score
        s = self.expression(x)
        if d is not None and "idx" in d:
            print(self.__str__(), "input", x[d["idx"]], "out", s[d["idx"]])
        return s
    
    def __str__(self):
        return "AP%d"%(self.apid) if self.comment is None else self.comment 

class And(STLFormula):
    def __init__(self, lhs, rhs):
        super(And, self).__init__(lhs=lhs, rhs=rhs, operator={"symbol": "&", "word": "AND"})

    def __call__(self, x, tau, d=None):
        s = softmin_pairs(self.lhs(x, tau, d), self.rhs(x, tau, d), tau, d)
        if d is not None and "idx" in d:
            print("And", "input", x[d["idx"], :], "output", s[d["idx"]])
        return s

class ListAnd(STLFormula):
    def __init__(self, lists):
        super(ListAnd, self).__init__(lists=lists, operator={"symbol": "&", "word": "AND"})

    def __call__(self, x, tau, d=None, full=False):
        v = [ap(x, tau, d) for ap in self.lists]
    
        v = torch.stack(v, dim=1)
        if d is not None and "idx" in d:
            print("And", "input", x[d["idx"], :], "output", s[d["idx"]])
        s = softmin(v, tau, d)[:, 0]

        if full:
            return s, v
        else:
            return s

class Or(STLFormula):
    def __init__(self, lhs, rhs):
        super(Or, self).__init__(lhs=lhs, rhs=rhs, operator={"symbol": "|", "word": "OR"})
    
    def __call__(self, x, tau, d=None):
        v1 = self.lhs(x, tau, d)
        v2 = self.rhs(x, tau, d)
        s = softmax_pairs(v1, v2, tau, d)
        if d is not None and "idx" in d:
            print("Or", "input", x[d["idx"], :], "lhs",v1[d["idx"]], "rhs", v2[d["idx"]], "output", s[d["idx"]])
        return s

class Not(STLFormula):
    def __init__(self, node):
        super(Not, self).__init__(node=node, operator={"symbol": "¬", "word": "NOT"})
    
    def __call__(self, x, tau, d=None):
        return -self.node(x, tau, d)

class Imply(STLFormula):
    def __init__(self, lhs, rhs):
        super(Imply, self).__init__(lhs=lhs, rhs=rhs, operator={"symbol": "->", "word": "IMPLY"})
        self.eval = Or(Not(self.lhs), self.rhs)
    
    def __call__(self, x, tau, d=None):
        s = self.eval(x, tau, d)
        if d is not None and "idx" in d:
            print("Imply", "input", x[d["idx"], :], "output", s[d["idx"]])
        return s

class Eventually(STLFormula):
    def __init__(self, ts, te, node):
        super(Eventually, self).__init__(ts=ts, te=te, node=node, operator={"symbol":"♢", "word":"EVENTUALLY"})
    
    def __call__(self, x, tau, d=None):
        s = self.node(x, tau, d)
        T = s.shape[1]
        scores = [softmax(s[:, clip(t+self.ts, 0, T): clip(t+self.te, 0, T)], tau, d) for t in range(T)]
        scores = torch.cat(scores, dim=-1)
        if d is not None and "idx" in d:
            print("Eventually", self.ts, self.te, "input", x[d["idx"], :], "output", scores[d["idx"]])
        return scores

class Always(STLFormula):
    def __init__(self, ts, te, node):
        super(Always, self).__init__(ts=ts, te=te, node=node, operator={"symbol": "◻", "word": "ALWAYS"})
    
    def __call__(self, x, tau, d=None):
        s = self.node(x, tau, d)
        T = s.shape[1]
        scores = [softmin(s[:, clip(t+self.ts, 0, T): clip(t+self.te, 0, T)], tau, d) for t in range(T)]
        scores = torch.cat(scores, dim=-1)
        
        if d is not None and "idx" in d:
            print("Always", self.ts, self.te, "input", x[d["idx"], :], "s", s, "output", scores[d["idx"]])
        return scores

class Once(STLFormula):
    def __init__(self, ts, te, node):
        super(Once, self).__init__(ts=ts, te=te, node=node, operator={"symbol":"O", "word":"ONCE"})
        assert ts<0 and te>=ts and te<=0

    def __call__(self, x, tau, d=None):
        s = self.node(x, tau, d)
        T = s.shape[1]
        scores = [softmax(s[:, clip(t+self.ts, 0, T): clip(t+self.te, 0, T)], tau, d) for t in range(T)]
        return torch.cat(scores, dim=-1)

class UntimedUntil(STLFormula):
    def __init__(self, lhs, rhs):
        super(UntimedUntil, self).__init__(lhs=lhs, rhs=rhs, operator={"symbol": "U", "word": "UNTIL"})
    
    def __call__(self, x, tau, d=None):
        ls = self.lhs(x, tau, d) # (n, t)
        rs = self.rhs(x, tau, d) # (n, t)
        inf_ls = -torch.logcumsumexp(-ls * tau, dim=1) / tau
        min_rs_inf_ls = softmin_pairs(rs, inf_ls, tau, d)
        scores = (torch.logcumsumexp(min_rs_inf_ls.flip(1) * tau, dim=1) / tau).flip(1)
        return scores

class Until(STLFormula):
    def __init__(self, ts, te, lhs, rhs):
        super(Until, self).__init__(ts=ts, te=te, lhs=lhs, rhs=rhs, operator={"symbol": "U", "word": "UNTIL"})
        if ts==0:
            self.eval = UntimedUntil(lhs, rhs)
        else:
            self.eval = And(Eventually(ts, te, rhs), Always(0, ts, UntimedUntil(lhs, rhs)))

    def __call__(self, x, tau, d=None):
        return self.eval(x, tau, d)
