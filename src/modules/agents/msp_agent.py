import torch.nn as nn
import torch.nn.functional as F
import torch

class MSPAgent(nn.Module):
    """
    MSP implement based on QMIX
    """
    def __init__(self, input_shape, args):
        super(MSPAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_experts = args.n_experts
        self.n_actions = args.n_actions

        self.fc = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        self.experts = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.rnn_hidden_dim, args.n_actions))
             for _ in range(args.n_experts)])

        self.gate = nn.Linear(input_shape, args.n_experts)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        #inputs shape:[b*a,input_shape]
        # expert calculate
        x = F.relu(self.fc(inputs)) #[b*a,rnn_hidden_dim]
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in) #[b*a,rnn_hidden_dim]
        experts_out = torch.stack([expert(h) for expert in self.experts],dim=0) #[n_experts,b*a,n_actions]
        # gate calculate
        weight = F.sigmoid(self.gate(inputs)) #[b*a,n_experts]     
        weight = weight.T.unsqueeze(-1) #[n_experts,b*a,1]
        weight = weight.expand(experts_out.size()) #[n_experts,b*a,n_actions]
        # merge
        final_q = (experts_out * weight).mean(dim=0) #[b*a,n_actions]     

        return final_q.view(b, a, -1), h.view(b, a, -1), experts_out.transpose(0,1)

