# frozen_string_literal: true

class EmpsController < ApplicationController
  PLT = Matplotlib::Pyplot

  def index; end

  def create
    @dataset = Pandas.read_csv(emp_params[:data].path, sep: ' ', header: nil)
    @values = remove_anomalies(@dataset.values.to_a.flatten.map(&:to_f))
    @count = @values.count
    @sorted_values = @values.sort
    @statistics = @sorted_values.tally.map do |value, count|
      {
        x: value, n: count, p: count.to_f / @count,
        empir: @values.count { |e| e <= value }.to_f / @count
      }
    end
    @arrow_chart = arrow_chart
    @m = params[:classes_amount].presence&.to_i || (1 + (3.32 * Math.log10(@count))).round
    @h = (@sorted_values.last - @sorted_values.first) / @m
    @classes = Array.new(@m) do |index|
      klass = [@sorted_values.first + (index * @h), @sorted_values.first + (index.next * @h)]
      index == @m.pred ? (klass.first..klass.last) : (klass.first...klass.last)
    end
    @class_statistics = @classes.index_with do |klass|
      values = @sorted_values.select { |value| klass.include?(value) }
      { values: values, p: values.count.to_f / @count, n: values.count,
        empir: @values.count { |e| e <= klass.last }.to_f / @count }
    end
    @window = params[:window_size].presence&.to_f || (Numpy.std(@values) * (@count**-0.2))
    @classes_gistogram = classes_gistogram
    @mean = @values.sum.to_f / @count
    @median = (@sorted_values[(@sorted_values.size.pred / 2)] + @sorted_values[@sorted_values.size / 2]).to_f / 2
    @global_statistics = global_statistics
    @probability_distribution = probability_distribution
    @new_classes_gistogram = new_classes_gistogram
    @empir_gist = empir_gist
  end

  private

  def empir_gist
    PLT.scatter(@statistics.pluck(:x), @statistics.pluck(:empir))
    PLT.ylabel('p')
    PLT.xlabel('x')
    plot_image
  end

  def new_classes_gistogram
    PLT.ylabel('p')
    PLT.xlabel('x')
    PLT.hist(@values, bins: @m, weights: Array.new(@count) { 1.0 / @count })
    k = ->(u) { (1 / Math.sqrt(2 * Math::PI)) * Math.exp(-(u**2).to_f / 2) }
    ys = @sorted_values.map do |value|
      1.0 / (@values.size * @window) * @values.sum { |value_i| k[(value - value_i) / @window] } * @h
    end
    PLT.plot(@sorted_values, ys)
    PLT.plot(@sorted_values, @sorted_values.map { |x| x > @a && x <= @b ? 1.0 / (@b - @a) * @h : 0 })
    plot_image
  end

  def probability_distribution
    modifier = Math.sqrt(3 * ((@values.sum { |value| value**2 }.to_f / @count) - (@mean**2)))
    @a = @mean - modifier
    @b = @mean + modifier
    cov = (@a + @b) * ((@b - @a)**2).to_f / 12 / @count
    dx = ((@b - @a)**2).to_f / 12 / @count
    dx2 = (1.0 / 180 / @count) * (((@b - @a)**4) + (15 * ((@a + @b)**2) * ((@b - @a)**2)))
    dh1dx = 1 + (3 * (@a + @b).to_f / (@b - @a))
    dh2dx = 1 - (3 * (@a + @b).to_f / (@b - @a))
    dh1dx2 = - 3.0 / (@b - @a)
    dh2dx2 = 3.0 / (@b - @a)
    da = ((dh1dx**2) * dx) + ((dh1dx2**2) * dx2) + (2 * dh1dx * dh1dx2 * cov)
    db = ((dh2dx**2) * dx) + ((dh2dx2**2) * dx2) + (2 * dh2dx * dh2dx2 * cov)
    @std_a = Math.sqrt(da)
    @std_b = Math.sqrt(db)
    @interval_a = (@a - (2 * @std_a)..@a + (2 * @std_a))
    @interval_b = (@b - (2 * @std_b)..@b + (2 * @std_b))
    f = lambda { |x|
      if x < @a then 0
      elsif x >= @b then 1
      else
        (x - @a).to_f / (@b - @a)
      end
    }
    PLT.scatter(@values, @values.map(&f))
    PLT.ylabel = 'Empir'
    PLT.xlabel = 'x'
    # PLT.plot(@values.minmax, @values.minmax.map(&f), color: 'orange')
    plot_image
  end

  def remove_anomalies(values)
    mean = values.sum.to_f / values.count
    std = Math.sqrt(values.sum { |value| (value - mean)**2 }.to_f / values.count.pred).to_f
    @anomalies_a = mean - (2 * std)
    @anomalies_b = mean + (2 * std)
    values = if ActiveModel::Type::Boolean.new.cast(params[:remove_anomalies])
               values.select { |value| (@anomalies_a...@anomalies_b).cover?(value) }
             else
               values
             end
    PLT.scatter(values.count.times.to_a, values)
    [@anomalies_a, @anomalies_b].each { |value| PLT.plot(Array.new(values.count) { |index| index }, Array.new(values.count) { value }) }
    PLT.ylabel('x')
    PLT.xlabel('n')
    @anomalies_plot = plot_image
    values
  end

  def global_statistics
    std = Math.sqrt(@values.sum { |value| (value - @mean)**2 }.to_f / @count.pred).to_f
    std_moved = Math.sqrt(@values.sum { |value| (value - @mean)**2 }.to_f / @count)
    excess_moved = (@values.sum { |value| (value - @mean)**4 }.to_f / @count / (std_moved**4)) - 3
    mean_std = std / Math.sqrt(@count)
    std_std = std / Math.sqrt(2 * @count)
    assymetry = (Math.sqrt(@count * @count.pred) / (@count - 2)) * (@values.sum { |value| (value - @mean)**3 }.to_f / @count / (std_moved**3))
    assymetry_std = Math.sqrt((6 * @count * @count.pred).to_f / ((@count - 2) * @count.next * (@count + 3)))
    excess = ((@count**2).pred / (@count - 2) / (@count - 3)) * (excess_moved + (6.0 / @count.next))
    excess_std = Math.sqrt((24 * @count * (@count.pred**2)).to_f / ((@count - 2) * (@count - 3) * (@count + 3) * (@count + 5)))
    {
      mean: { value: @mean, std: mean_std, ninety_five: (@mean - (2 * mean_std)..@mean + mean_std) },
      median: {
        value: @median,
        ninety_five: (@sorted_values[(@count.to_f / 2) - (1.96 * (Math.sqrt(@count).to_f / 2 )).round]..@sorted_values[(@count.to_f / 2) + 1 + (1.96 * (Math.sqrt(@count).to_f / 2 )).round]) },
      std: { value: std, std: std_std, ninety_five: (std - (2 * std_std)..std + (2 * std_std)) },
      assymetry: {
        value: assymetry, std: assymetry_std,
        ninety_five: (assymetry - (2 * assymetry_std)..assymetry + (2 * assymetry_std))
      },
      excess: {
        value: excess, std: excess_std,
        ninety_five: (excess - (2 * excess_std)..excess + (2 * excess_std))
      },
      counter_excess: { value: 1.0 / Math.sqrt(excess_moved + 3), ninety_five: nil },
      minimum: { value: @sorted_values.first },
      maximum: { value: @sorted_values.last }
    }
  end

  def arrow_chart
    @statistics[0..-2].each_with_index do |statistic, index|
      PLT.arrow(statistic[:x], statistic[:empir], @statistics[index.next][:x] - statistic[:x], 0,
                width: 0.02, color: 'orange')
    end
    PLT.ylabel = 'Empir'
    PLT.xlabel = 'x'
    plot_image
  end

  def classes_gistogram
    PLT.ylabel = 'p'
    PLT.xlabel = 'x'
    PLT.hist(@values, bins: @m, weights: Array.new(@count) { 1.0 / @count })
    k = ->(u) { (1 / Math.sqrt(2 * Math::PI)) * Math.exp(-(u**2).to_f / 2) }
    ys = @sorted_values.map do |value|
      1.0 / (@values.size * @window) * @values.sum { |value_i| k[(value - value_i) / @window] } * @h
    end
    PLT.plot(@sorted_values, ys)
    plot_image
  end

  def emp_params
    params.permit(:data, :window_size, :remove_anomalies, :classes_amount)
  end

  def plot_image
    filename = Rails.root.join("tmp/#{SecureRandom.hex}")
    PLT.savefig(File.new(filename, 'wb'))
    "data:image/png;base64,#{Base64.strict_encode64(File.read(filename))}"
  ensure
    File.delete(filename)
    PLT.clf
  end
end
