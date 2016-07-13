import theano
import theano.tensor as T
import numpy as np


class ThreeLayerTiedWeightsAttendedLSTMLayer(object):

    def __init__(self,random_state,input,input_dim,output_dim,outer_output_dim,bptt_truncate=-1,layer_id="_0"):
        self.input = input
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.outer_output_dim = outer_output_dim
        self.random_state = random_state
        self.layer_id = layer_id
        self.initialize_params()
        self.bptt_truncate = bptt_truncate



        def forward_step(x_t, prev_state, prev_content, prev_state_2, prev_content_2, prev_state_3, prev_content_3):
            input_gate = T.nnet.hard_sigmoid(T.dot(( self.U_input),x_t) + T.dot(self.W_input,prev_state) + self.bias_input)
            forget_gate = T.nnet.hard_sigmoid(T.dot(( self.U_forget),x_t) + T.dot(self.W_forget,prev_state)+ self.bias_forget)
            output_gate = T.nnet.hard_sigmoid(T.dot((self.U_output),x_t) + T.dot(self.W_output,prev_state)+ self.bias_output)

            stabilized_input = T.tanh(T.dot((self.U),x_t) + T.dot(self.W,prev_state) + self.bias)
            c = forget_gate * prev_content + input_gate * stabilized_input
            s1 = output_gate * T.tanh(c)

            input_gate_2 = T.nnet.hard_sigmoid(
                T.dot((self.U_input), s1) + T.dot(self.W_input, prev_state_2) + self.bias_input)
            forget_gate_2 = T.nnet.hard_sigmoid(
                T.dot((self.U_forget), s1) + T.dot(self.W_forget, prev_state_2) + self.bias_forget)
            output_gate_2 = T.nnet.hard_sigmoid(
                T.dot((self.U_output), s1) + T.dot(self.W_output, prev_state_2) + self.bias_output)

            stabilized_input_2 = T.tanh(T.dot((self.U), s1) + T.dot(self.W, prev_state_2) + self.bias)
            c2 = forget_gate_2 * prev_content_2 + input_gate_2 * stabilized_input_2
            s2 = output_gate_2 * T.tanh(c2)

            input_gate_3 = T.nnet.hard_sigmoid(
                T.dot((self.U_input), s2) + T.dot(self.W_input, prev_state_3) + self.bias_input)
            forget_gate_3 = T.nnet.hard_sigmoid(
                T.dot((self.U_forget), s2) + T.dot(self.W_forget, prev_state_3) + self.bias_forget)
            output_gate_3 = T.nnet.hard_sigmoid(
                T.dot((self.U_output), s2) + T.dot(self.W_output, prev_state_3) + self.bias_output)

            stabilized_input_3 = T.tanh(T.dot((self.U), s2) + T.dot(self.W, prev_state_3) + self.bias)
            c3 = forget_gate_3 * prev_content_3 + input_gate_3 * stabilized_input_3
            s3 = output_gate_3 * T.tanh(c3)


            o = T.nnet.sigmoid(T.dot(self.O_w,s3)+self.O_bias)

            return [o,s1,c,s2,c2,s3,c3,input_gate, forget_gate, output_gate]

        [self.output,self.hidden_state,self.memory_content,self.hidden_state_2,self.memory_content_2,self.hidden_state_3,self.memory_content_3,self.input_gate, self.forget_gate,
         self.output_gate] , updates = theano.scan(
            forward_step,
            sequences=[self.input],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX))
                          , dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.output_dim,dtype=theano.config.floatX))
                         , dict(initial=T.zeros(self.output_dim, dtype=theano.config.floatX)),
                          dict(initial=T.zeros(self.output_dim, dtype=theano.config.floatX))
                          , None, None, None
                          ])

    def initialize_params(self):
        U_input = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        U_forget = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        U_output = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        W_input = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_forget = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        W_output = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        U = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.input_dim + self.output_dim)),
                                     (self.output_dim, self.input_dim))
            , dtype=theano.config.floatX)

        W = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                     (self.output_dim, self.output_dim))
            , dtype=theano.config.floatX)

        bias_input = np.zeros(self.output_dim, dtype=theano.config.floatX)
        bias_forget = np.zeros(self.output_dim, dtype=theano.config.floatX)
        bias_output = np.zeros(self.output_dim, dtype=theano.config.floatX)
        bias = np.zeros(self.output_dim, dtype=theano.config.floatX)

        self.W = theano.shared(value=W, name="W" + self.layer_id, borrow="True")
        self.U = theano.shared(value=U, name="U" + self.layer_id, borrow="True")
        self.bias = theano.shared(value=bias, name="bias", borrow="True")

        self.W_input = theano.shared(value=W_input, name="W_input" + self.layer_id, borrow="True")
        self.U_input = theano.shared(value=U_input, name="U_input" + self.layer_id, borrow="True")
        self.bias_input = theano.shared(value=bias_input, name="bias_input" + self.layer_id, borrow="True")

        self.W_output = theano.shared(value=W_output, name="W_output" + self.layer_id, borrow="True")
        self.U_output = theano.shared(value=U_output, name="U_output" + self.layer_id, borrow="True")
        self.bias_output = theano.shared(value=bias_output, name="bias_output" + self.layer_id, borrow="True")

        self.W_forget = theano.shared(value=W_forget, name="W_forget" + self.layer_id, borrow="True")
        self.U_forget = theano.shared(value=U_forget, name="U_forget" + self.layer_id, borrow="True")
        self.bias_forget = theano.shared(value=bias_forget, name="bias_forget" + self.layer_id, borrow="True")


        O_w = np.asarray(
            self.random_state.normal(-np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      np.sqrt(6.0 / (self.output_dim + self.output_dim)),
                                      (self.outer_output_dim,self.output_dim))
            , dtype=theano.config.floatX)

        O_bias = np.zeros(self.outer_output_dim, dtype=theano.config.floatX)

        self.O_w = theano.shared(value=O_w, name="O_w"+self.layer_id, borrow="True")
        self.O_bias = theano.shared(value=O_bias, name="O_bias"+self.layer_id, borrow="True")

        self.params = [self.U_input, self.U_forget, self.U_output, self.W_input, self.W_forget, self.W_output,
                       self.bias_input, self.bias_forget, self.bias_output, self.U, self.W, self.bias, ]

        self.output_params = [self.O_w,self.O_bias]