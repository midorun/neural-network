class NeuralNetwork {
  constructor(inputnodes, hiddennodes, outputnodes, learningrate, wih, who) {
    this.inputnodes = inputnodes;
    this.hiddennodes = hiddennodes;
    this.outputnodes = outputnodes;
    this.learningrate = learningrate;

    this.wih =
      wih ||
      math.subtract(math.matrix(math.random([hiddennodes, inputnodes])), 0.5);
    this.who =
      who ||
      math.subtract(math.matrix(math.random([outputnodes, hiddennodes])), 0.5);

    this.act = (matrix) => math.map(matrix, (x) => 1 / (1 + Math.exp(-x)));
  }

  static normalizeData = (data) => {
    return data.map((e) => (e / 255) * 0.99 + 0.01);
  };

  cache = { loss: [] };

  forward = (input) => {
    const wih = this.wih;
    const who = this.who;
    const act = this.act;

    input = math.transpose(math.matrix([input]));

    const h_in = math.evaluate("wih * input", { wih, input });
    const h_out = act(h_in);

    const o_in = math.evaluate("who * h_out", { who, h_out });
    const actual = act(o_in);

    this.cache.input = input;
    this.cache.h_out = h_out;
    this.cache.actual = actual;

    return actual;
  };

  backward = (target) => {
    const who = this.who;
    const input = this.cache.input;
    const h_out = this.cache.h_out;
    const actual = this.cache.actual;

    target = math.transpose(math.matrix([target]));

    const dEdA = math.subtract(target, actual);

    const o_dAdZ = math.evaluate("actual .* (1 - actual)", {
      actual,
    });

    const dwho = math.evaluate("(dEdA .* o_dAdZ) * h_out'", {
      dEdA,
      o_dAdZ,
      h_out,
    });

    const h_err = math.evaluate("who' * (dEdA .* o_dAdZ)", {
      who,
      dEdA,
      o_dAdZ,
    });

    const h_dAdZ = math.evaluate("h_out .* (1 - h_out)", {
      h_out,
    });

    const dwih = math.evaluate("(h_err .* h_dAdZ) * input'", {
      h_err,
      h_dAdZ,
      input,
    });

    this.cache.dwih = dwih;
    this.cache.dwho = dwho;
    this.cache.loss.push(math.sum(math.square(dEdA)));
  };

  update = () => {
    const wih = this.wih;
    const who = this.who;
    const dwih = this.cache.dwih;
    const dwho = this.cache.dwho;
    const r = this.learningrate;

    this.wih = math.evaluate("wih + (r .* dwih)", { wih, r, dwih });
    this.who = math.evaluate("who + (r .* dwho)", { who, r, dwho });
  };

  predict = (input) => {
    return this.forward(input);
  };

  train = (input, target) => {
    this.forward(input);
    this.backward(target);
    this.update();
  };
}
