class EmpThreesController < ApplicationController
  PLT = Matplotlib::Pyplot
  ALPHA = 0.05

  def show; end

  def create
    @dataset = Pandas.read_csv(emp_params[:data]&.path, sep: ' ', header: nil)&.values&.to_a
    (@data = emp_params[:independent].to_i.zero? ? dependent : independent) if @dataset

    @first_task = first_task
    @second_mortality_task = second_mortality_task
    @second_hardness_task = second_hardness_task
    @third_task = third_task
    @fourth_task = fourth_task
  end

  private

  def first_task
    df = Pandas.read_csv('readingspeed.txt').values.to_a.flatten
    dra = df.select { |x| x.match?(/DRA/) }.map(&:to_f)
    sc = df.select { |x| x.match?(/SC/) }.map(&:to_f)
    p_dra = ::ScipyStats.normaltest(dra).pvalue
    p_sc = ::ScipyStats.normaltest(sc).pvalue
    dra_stat = mean_confidence_interval(dra)
    sc_stat = mean_confidence_interval(sc)
    statistics = { dra: { image: plot(dra.sort),
                          mean: dra_stat[:mean],
                          interval: dra_stat[:interval] },
                   sc: { image: plot(sc.sort, color: 'green'),
                         mean: sc_stat[:mean],
                         interval: sc_stat[:interval] } }
    { p_dra: p_dra,
      p_sc: p_sc,
      statistics: statistics,
      f_test: f_test(dra, sc),
      welch_two_sample_t_test: welch_two_sample_t_test(dra, sc) }
  end

  def second_mortality_task
    df = Pandas.read_csv('water.txt').values.to_a.flatten
    south = df.select { |x| x.match?(/South/) }.map { |x| x.match(/\d+/).to_s.to_f }
    north = df.select { |x| x.match?(/North/) }.map { |x| x.match(/\d+/).to_s.to_f }
    p_south = ::ScipyStats.normaltest(south).pvalue
    p_north = ::ScipyStats.normaltest(north).pvalue
    south_stat = mean_confidence_interval(south)
    north_stat = mean_confidence_interval(north)
    statistics = { south: { image: plot(south.sort),
                            mean: south_stat[:mean],
                            interval: south_stat[:interval] },
                   north: { image: plot(north.sort, color: 'green'),
                            mean: north_stat[:mean],
                            interval: north_stat[:interval] } }
    { p_south: p_south,
      p_north: p_north,
      statistics: statistics,
      f_test: f_test(south, north),
      two_sample_t_test: two_sample_t_test(south, north) }
  end

  def second_hardness_task
    df = Pandas.read_csv('water.txt').values.to_a.flatten
    south = df.select { |x| x.match?(/South/) }.map { |x| x.split("\t").last.to_f }
    north = df.select { |x| x.match?(/North/) }.map { |x| x.split("\t").last.to_f }
    p_south = ::ScipyStats.normaltest(south).pvalue
    p_north = ::ScipyStats.normaltest(north).pvalue
    south_stat_mean = mean_confidence_interval(south)
    north_stat_mean = mean_confidence_interval(north)
    south_stat_median = median_confidence_interval(south)
    north_stat_median = median_confidence_interval(north)
    statistics = { south: { image: plot(south.sort),
                            mean: south_stat_mean[:mean],
                            mean_interval: south_stat_mean[:interval],
                            median: south_stat_median[:median],
                            median_interval: south_stat_mean[:interval] },
                   north: { image: plot(north.sort, color: 'green'),
                            mean: north_stat_mean[:mean],
                            mean_interval: north_stat_mean[:interval],
                            median: north_stat_median[:median],
                            median_interval: north_stat_median[:interval] } }
    { p_south: p_south,
      p_north: p_north,
      statistics: statistics,
      wilcoxon_rank_sums_test: wilcoxon_rank_sums_test(south, north) }
  end

  def third_task
    df = Pandas.read_csv('ADHD.txt').values.to_a.flatten
    placebo = df.map { |x| x.split.first.to_f }
    methylphenidate = df.map { |x| x.split.last.to_f }
    df_diff = placebo.map.with_index { |x, i| x - methylphenidate[i] }
    p_diff = ::ScipyStats.normaltest(df_diff).pvalue
    stat_diff = mean_confidence_interval(df_diff)
    stat_placebo = mean_confidence_interval(placebo)
    stat_methylphenidate = mean_confidence_interval(methylphenidate)
    statistics = { diff: { image: plot(df_diff.sort),
                           mean: stat_diff[:mean],
                           interval: stat_diff[:interval] },
                   placebo: { mean: stat_placebo[:mean],
                              interval: stat_placebo[:interval] },
                   methylphenidate: { mean: stat_methylphenidate[:mean],
                                      interval: stat_methylphenidate[:interval] } }
    { p_diff: p_diff,
      statistics: statistics,
      f_test: f_test(placebo, methylphenidate),
      paried_t_test: paried_t_test(placebo, methylphenidate) }
  end

  def fourth_task
    df = Pandas.read_csv('wines.txt').values.to_a.flatten
    shardone = df.map { |x| x.split[1].to_f }
    kaberne = df.map { |x| x.split.last.to_f }
    p_shardone = ::ScipyStats.normaltest(shardone).pvalue
    p_kaberne = ::ScipyStats.normaltest(kaberne).pvalue
    stat_shardone_mean = mean_confidence_interval(shardone)
    stat_kaberne_mean = mean_confidence_interval(kaberne)
    stat_shardone_median = median_confidence_interval(shardone)
    stat_kaberne_median = median_confidence_interval(kaberne)
    statistics = { shardone: { image: plot(shardone.sort),
                               mean: stat_shardone_mean[:mean],
                               mean_interval: stat_shardone_mean[:interval],
                               median: stat_shardone_median[:median],
                               median_interval: stat_shardone_median[:interval] },
                   kaberne: { image: plot(kaberne.sort, color: 'green'),
                              mean: stat_kaberne_mean[:mean],
                              mean_interval: stat_kaberne_mean[:interval],
                              median: stat_kaberne_median[:median],
                              median_interval: stat_kaberne_median[:interval] } }
    { p_shardone: p_shardone,
      p_kaberne: p_kaberne,
      statistics: statistics,
      f_test: f_test(shardone, kaberne),
      paried_t_test: paried_t_test(shardone, kaberne) }
  end

  def independent
    first_df = @dataset.map(&:first)
    second_df = @dataset.map(&:last)
    p_first = ::ScipyStats.normaltest(first_df).pvalue
    p_second = ::ScipyStats.normaltest(second_df).pvalue
    first_stat_mean = mean_confidence_interval(first_df)
    second_stat_mean = mean_confidence_interval(second_df)
    f_test = f_test(first_df, second_df)
    if p_first >= 0.05 && p_second >= 0.05
      statistics = { first: { image: plot(first_df.sort),
                              mean: first_stat_mean[:mean],
                              mean_interval: first_stat_mean[:interval] },
                     second: { image: plot(second_df.sort),
                               mean: second_stat_mean[:mean],
                               mean_interval: second_stat_mean[:interval] } }
      math_expect = f_test[:answer] ? two_sample_t_test(first_df, second_df) : welch_two_sample_t_test(first_df, second_df)
    else
      first_stat_median = median_confidence_interval(first_df)
      second_stat_median = median_confidence_interval(second_df)
      statistics = { first: { image: plot(first_df.sort),
                              mean: first_stat_mean[:mean],
                              mean_interval: first_stat_mean[:interval],
                              median: first_stat_median[:median],
                              median_interval: first_stat_median[:interval] },
                     second: { image: plot(second_df.sort),
                               mean: second_stat_mean[:mean],
                               mean_interval: second_stat_mean[:interval],
                               median: second_stat_median[:median],
                               median_interval: second_stat_median[:interval] } }
      math_expect = wilcoxon_rank_sums_test(first_df, second_df)
    end
    { p_first: p_first,
      p_second: p_second,
      statistics: statistics,
      f_test: f_test,
      math_expect: math_expect }
  end

  def dependent
    first_df = @dataset.map(&:first)
    second_df = @dataset.map(&:last)
    df = first_df.map.with_index { |x, i| x - second_df[i] }
    p_df = ::ScipyStats.normaltest(df).pvalue
    stat_mean = mean_confidence_interval(df)
    f_test = f_test(first_df, second_df)
    if p_df >= 0.05
      statistics = { first: { image: plot(df.sort),
                              mean: stat_mean[:mean],
                              mean_interval: stat_mean[:interval] } }
      math_expect = paried_t_test(first_df, second_df)
    else
      stat_median = median_confidence_interval(first_df)
      statistics = { first: { image: plot(df.sort),
                              mean: stat_mean[:mean],
                              mean_interval: stat_mean[:interval],
                              median: stat_median[:median],
                              median_interval: stat_median[:interval] } }
      math_expect = wilcoxon_signed_rank_test(first_df, second_df)
    end
    { p_diff: p_df,
      statistics: statistics,
      f_test: f_test,
      math_expect: math_expect }
  end

  def mean_confidence_interval(df)
    mean = df.sum.to_f / df.count
    std = Numpy.std(df)
    t = ScipyStats.t(df.count.pred).ppf(1 - (ALPHA / 2))
    interval = [mean - (std * t / 4), mean + (std * t / 4)]
    { mean: mean,
      interval: (interval.first..interval.last) }
  end

  def median_confidence_interval(df)
    n = df.count
    range = ScipyStats.norm.ppf(1 - (ALPHA / 2)) * Math.sqrt(n) / 2

    j = (n / 2) - 1 - range
    k = (n / 2) + range

    sorted_df = df.sort
    median = (sorted_df[(sorted_df.size.pred / 2)] + sorted_df[sorted_df.size / 2]).to_f / 2
    { median: median,
      interval: (sorted_df[j.floor]..sorted_df[k.floor]) }
  end

  def plot(df, color: 'blue')
    PLT.scatter(df.count.times.to_a, df, color: color)
    plot_image
  end

  def f_test(first_df, second_df)
    s1 = dispersia(first_df)
    s2 = dispersia(second_df)
    f = s1 / s2
    if f <= 1
      p = 2 * ScipyStats.f(first_df.count, second_df.count).cdf(f)
    else
      p = 2 * (1 - ScipyStats.f(first_df.count, second_df.count).cdf(f))
    end
    { f: f,
      p: p,
      answer: p >= ALPHA }
  end

  def dispersia(df)
    n = df.count
    mean = df.sum.to_f / n
    df.sum { |x| (x - mean)**2 }.to_f / n.pred
  end

  def paried_t_test(first_df, second_df)
    z = first_df.map.with_index { |x, i| x - second_df[i] }
    mean = z.sum.to_f / z.count
    s = Math.sqrt(dispersia(z))
    t = mean * Math.sqrt(z.count) / s
    p = 2 * (1 - ScipyStats.t(z.count.pred).cdf(t.abs))
    { t: t,
      p: p,
      answer: p >= ALPHA }
  end

  def two_sample_t_test(first_df, second_df)
    n1 = first_df.count
    n2 = second_df.count
    s1 = dispersia(first_df)
    s2 = dispersia(second_df)
    mean1 = first_df.sum.to_f / first_df.count
    mean2 = second_df.sum.to_f / second_df.count

    s = (((n1 - 1) * s1) + ((n2 - 1) * s2)) / (n1 - 1 + n2 - 1)
    t = (mean1 - mean2) / Math.sqrt((s / n1) + (s / n2))
    p = 2 * (1 - ScipyStats.t(n1 - 1 + n2 - 1).cdf(t.abs))
    { t: t,
      p: p,
      answer: p >= ALPHA }
  end

  def welch_two_sample_t_test(first_df, second_df)
    n1 = first_df.count
    n2 = second_df.count
    s1 = dispersia(first_df)
    s2 = dispersia(second_df)
    mean1 = first_df.sum.to_f / first_df.count
    mean2 = second_df.sum.to_f / second_df.count

    # s = (((n1 - 1) * s1) + ((n2 - 1) * s2)) / (n1 - 1 + n2 - 1)
    t = (mean1 - mean2) / Math.sqrt((s1 / n1) + (s2 / n2))
    v = (((s1 / n1) + (s2 / n2))**2) / ((((s1 / n1)**2) / (n1 - 1)) + (((s2 / n2)**2) / (n2 - 1)))
    p = 2 * (1 - ScipyStats.t(v).cdf(t.abs))
    { t: t,
      p: p,
      answer: p >= ALPHA }
  end

  def wilcoxon_signed_rank_test(first_df, second_df)
    z = first_df.map.with_index { |x, i| x - second_df[i] }
    z = z.reject(&:zero?)
    # s = z.map { |x| x > 0 ? 1 : 0 }
    z_sort = z.sort
    s_sort = z_sort.map { |x| x.positive? ? 1 : 0 }
    ranks = z_sort.map.with_index { |_, i| i.next }

    new_ranks = []
    ranks.map.with_index do |_, i|
      count = z_sort.count { |x| x == z_sort[i] }
      sum_ranks = []
      count.times { |time| sum_ranks.append(ranks[i + time]) }
      new_rank = sum_ranks.sum.to_f / count
      count.times { |time| new_ranks[i + time] = new_rank }
    end

    t = s_sort.map.with_index { |s, i| s * new_ranks[i] }.sum

    et = (z_sort.count * (z_sort.count + 1)).to_f / 4
    dt = (z_sort.count * (z_sort.count + 1) * ((2 * z_sort.count) + 1)).to_f / 24

    u = (t - et).to_f / Math.sqrt(dt)
    p = 2 * (1 - ScipyStats.norm.cdf(u.abs))

    { t: t,
      p: p,
      answer: p >= ALPHA }
  end

  def wilcoxon_rank_sums_test(first_df, second_df)
    df = (first_df + second_df).sort
    ranks = df.map.with_index { |_, i| i.next }

    new_ranks = []
    k = -1
    ranks.map.with_index do |_, i|
      next if i <= k

      count = df.count { |x| x == df[i] }
      sum_ranks = []
      count.times { |time| sum_ranks.append(ranks[i + time]) }
      new_rank = sum_ranks.sum.to_f / count
      count.times { |time| new_ranks[i + time] = new_rank }
      k += count
    end

    w = []
    df.each_with_index do |elem, i|
      # if first_df.count { |x| x == elem } <= df.count { |x| x == elem }
      if first_df.include?(elem) && df.include?(elem) && first_df.count { |x| x == elem } != w.count { |x| x == new_ranks[i.pred] }
        w.append(new_ranks[i])
      end
    end
    # w.count == first_df.count
    w = w.sum

    ew = (first_df.count * (df.count + 1)).to_f / 2
    dw = (first_df.count * second_df.count * (df.count + 1)).to_f / 12

    u = (w - ew).to_f / Math.sqrt(dw)
    p = 2 * (1 - ScipyStats.norm.cdf(u.abs))

    { w: w,
      u: u,
      p: p,
      answer: p >= ALPHA }
  end

  def emp_params
    params.permit(:data, :independent)
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
