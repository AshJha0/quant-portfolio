#include "fxvar/book.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fxvar {

namespace {

std::string upper(std::string s) {
  for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

}  // namespace

PairLegs split_pair(const std::string& pair) {
  if (pair.size() != 6 ||
      !std::all_of(pair.begin(), pair.end(), [](unsigned char c) {
        return std::isalpha(c) != 0;
      })) {
    throw std::invalid_argument("FX pair must be 6 letters like 'EURUSD', got '" +
                                pair + "'");
  }
  PairLegs legs{upper(pair.substr(0, 3)), upper(pair.substr(3, 3))};
  if (legs.base == legs.quote)
    throw std::invalid_argument("FX pair has identical legs: '" + pair + "'");
  return legs;
}

std::string fx_factor(const std::string& ccy) {
  const std::string c = upper(ccy);
  if (c == "USD")
    throw std::invalid_argument(
        "USD has no FX factor: its USD price is identically 1");
  return "FX:" + c;
}

std::string ir_factor(const std::string& ccy) { return "IR:" + upper(ccy); }

// ----------------------------------------------------------------- Market
Market::Market(std::map<std::string, double> spot_usd,
               std::map<std::string, double> rates) {
  for (auto& [c, s] : spot_usd) {
    if (!std::isfinite(s) || s <= 0.0) {
      std::ostringstream msg;
      msg << "spot_usd[" << c << "] must be a positive number, got " << s;
      throw std::invalid_argument(msg.str());
    }
    spot_usd_[upper(c)] = s;
  }
  if (auto it = spot_usd_.find("USD"); it != spot_usd_.end()) {
    if (std::abs(it->second - 1.0) > 1e-12)
      throw std::invalid_argument("spot_usd['USD'] must be 1.0 (USD per USD)");
  }
  spot_usd_["USD"] = 1.0;
  for (auto& [c, r] : rates) rates_[upper(c)] = r;
}

double Market::spot(const std::string& ccy) const {
  const auto it = spot_usd_.find(upper(ccy));
  if (it == spot_usd_.end())
    throw std::invalid_argument("no USD spot for currency '" + ccy +
                                "' in Market");
  return it->second;
}

double Market::rate(const std::string& ccy) const {
  const auto it = rates_.find(upper(ccy));
  if (it == rates_.end())
    throw std::invalid_argument("no interest rate for currency '" + ccy +
                                "' in Market");
  return it->second;
}

double Market::cross(const std::string& pair) const {
  const PairLegs legs = split_pair(pair);
  return spot(legs.base) / spot(legs.quote);
}

double Market::forward(const std::string& pair, double expiry) const {
  const PairLegs legs = split_pair(pair);
  return cross(pair) * std::exp((rate(legs.quote) - rate(legs.base)) * expiry);
}

// ------------------------------------------------------------------- Book
namespace {

void validate_position(const Position& p) {
  if (const auto* cash = std::get_if<CashPosition>(&p)) {
    if (cash->ccy.size() != 3)
      throw std::invalid_argument("cash ccy must be a 3-letter code, got '" +
                                  cash->ccy + "'");
  } else if (const auto* spot = std::get_if<SpotPosition>(&p)) {
    split_pair(spot->pair);
    if (spot->entry_rate && *spot->entry_rate <= 0.0)
      throw std::invalid_argument("entry_rate must be > 0");
  } else if (const auto* fwd = std::get_if<ForwardPosition>(&p)) {
    split_pair(fwd->pair);
    if (fwd->expiry < 0.0) throw std::invalid_argument("expiry must be >= 0");
    if (fwd->strike && *fwd->strike <= 0.0)
      throw std::invalid_argument("strike must be > 0");
  }
}

std::set<std::string> position_currencies(const Position& p) {
  if (const auto* cash = std::get_if<CashPosition>(&p))
    return {upper(std::string(cash->ccy))};
  if (const auto* spot = std::get_if<SpotPosition>(&p)) {
    const PairLegs legs = split_pair(spot->pair);
    return {legs.base, legs.quote};
  }
  const auto& fwd = std::get<ForwardPosition>(p);
  const PairLegs legs = split_pair(fwd.pair);
  return {legs.base, legs.quote};
}

std::string upper_copy(const std::string& s) { return upper(s); }

}  // namespace

Book::Book(std::vector<Position> positions, std::string base)
    : positions_(std::move(positions)), base_(upper_copy(base)) {
  for (const auto& p : positions_) validate_position(p);
}

std::vector<std::string> Book::currencies() const {
  std::set<std::string> ccys{base_};
  for (const auto& p : positions_) {
    const auto pc = position_currencies(p);
    ccys.insert(pc.begin(), pc.end());
  }
  return {ccys.begin(), ccys.end()};
}

std::vector<std::string> Book::factors() const {
  std::set<std::string> fx;
  std::set<std::string> ir;
  for (const auto& c : currencies())
    if (c != "USD") fx.insert(fx_factor(c));
  for (const auto& p : positions_) {
    if (const auto* fwd = std::get_if<ForwardPosition>(&p)) {
      const PairLegs legs = split_pair(fwd->pair);
      ir.insert(ir_factor(legs.base));
      ir.insert(ir_factor(legs.quote));
    }
  }
  std::vector<std::string> out(fx.begin(), fx.end());
  out.insert(out.end(), ir.begin(), ir.end());
  return out;
}

// ----------------------------------------------------------- CompiledBook
CompiledBook::CompiledBook(const Book& book, const Market& market) {
  if (book.empty())
    throw std::invalid_argument(
        "cannot compile an empty book: add positions before running VaR");
  factors_ = book.factors();
  base_ = book.base();
  s0_base_ = market.spot(base_);

  auto factor_index = [this](const std::string& f) -> int {
    for (std::size_t j = 0; j < factors_.size(); ++j)
      if (factors_[j] == f) return static_cast<int>(j);
    return -1;
  };
  base_fx_idx_ = (base_ == "USD") ? -1 : factor_index(fx_factor(base_));

  auto add_leg = [&](double amount, const std::string& ccy, double rate0,
                     double expiry) {
    const double df = (expiry > 0.0) ? std::exp(-rate0 * expiry) : 1.0;
    value0_.push_back(amount * market.spot(ccy) * df);
    fx_idx_.push_back(ccy == "USD" ? -1 : factor_index(fx_factor(ccy)));
    ir_idx_.push_back(expiry > 0.0 ? factor_index(ir_factor(ccy)) : -1);
    neg_expiry_.push_back(expiry > 0.0 ? -expiry : 0.0);
  };

  for (const auto& p : book.positions()) {
    if (const auto* cash = std::get_if<CashPosition>(&p)) {
      add_leg(cash->amount, upper_copy(cash->ccy), 0.0, 0.0);
    } else if (const auto* spot = std::get_if<SpotPosition>(&p)) {
      const PairLegs legs = split_pair(spot->pair);
      const double x0 =
          spot->entry_rate ? *spot->entry_rate : market.cross(spot->pair);
      add_leg(+spot->notional, legs.base, 0.0, 0.0);
      add_leg(-spot->notional * x0, legs.quote, 0.0, 0.0);
    } else {
      const auto& fwd = std::get<ForwardPosition>(p);
      const PairLegs legs = split_pair(fwd.pair);
      const double k =
          fwd.strike ? *fwd.strike : market.forward(fwd.pair, fwd.expiry);
      if (fwd.expiry > 0.0) {
        add_leg(+fwd.notional, legs.base, market.rate(legs.base), fwd.expiry);
        add_leg(-fwd.notional * k, legs.quote, market.rate(legs.quote),
                fwd.expiry);
      } else {  // expired forward = spot difference vs strike
        add_leg(+fwd.notional, legs.base, 0.0, 0.0);
        add_leg(-fwd.notional * k, legs.quote, 0.0, 0.0);
      }
    }
  }
  v0_usd_ = 0.0;
  for (double v : value0_) v0_usd_ += v;
}

double CompiledBook::value_usd(const double* shocks) const {
  double total = 0.0;
  const std::size_t n = value0_.size();
  for (std::size_t i = 0; i < n; ++i) {
    double shift = 0.0;
    if (fx_idx_[i] >= 0) shift += shocks[fx_idx_[i]];
    if (ir_idx_[i] >= 0) shift += neg_expiry_[i] * shocks[ir_idx_[i]];
    total += (shift == 0.0) ? value0_[i] : value0_[i] * std::exp(shift);
  }
  return total;
}

double CompiledBook::pnl(const double* shocks) const {
  const double v1 = value_usd(shocks);
  const double s1_base =
      (base_fx_idx_ >= 0) ? s0_base_ * std::exp(shocks[base_fx_idx_]) : s0_base_;
  return v1 / s1_base - v0_usd_ / s0_base_;
}

double CompiledBook::pnl(const std::map<std::string, double>& shocks) const {
  if (shocks.count("FX:USD"))
    throw std::invalid_argument(
        "shock to 'FX:USD' is not a valid factor: USD is the pivot (its USD "
        "price is identically 1). Shock the other leg(s).");
  std::vector<double> aligned(factors_.size(), 0.0);
  for (std::size_t j = 0; j < factors_.size(); ++j) {
    const auto it = shocks.find(factors_[j]);
    if (it != shocks.end()) aligned[j] = it->second;
  }
  return pnl(aligned.data());
}

std::vector<double> CompiledBook::pnl(const ReturnsMatrix& scenarios) const {
  // Map each leg's factor to a scenario column once, then stream rows.
  std::vector<int> fx_col(value0_.size(), -1), ir_col(value0_.size(), -1);
  std::vector<std::string> missing;
  std::vector<int> factor_col(factors_.size(), -1);
  for (std::size_t j = 0; j < factors_.size(); ++j) {
    factor_col[j] = scenarios.column_index(factors_[j]);
    if (factor_col[j] < 0) missing.push_back(factors_[j]);
  }
  if (!missing.empty()) {
    std::ostringstream msg;
    msg << "scenario matrix is missing required factor columns:";
    for (const auto& f : missing) msg << " " << f;
    throw std::invalid_argument(msg.str());
  }
  for (std::size_t i = 0; i < value0_.size(); ++i) {
    if (fx_idx_[i] >= 0) fx_col[i] = factor_col[fx_idx_[i]];
    if (ir_idx_[i] >= 0) ir_col[i] = factor_col[ir_idx_[i]];
  }
  const int base_col = (base_fx_idx_ >= 0) ? factor_col[base_fx_idx_] : -1;

  const std::size_t n_scen = scenarios.n_obs();
  const std::size_t n_legs = value0_.size();
  std::vector<double> out(n_scen);
  const double v0_base = v0_usd_ / s0_base_;
  for (std::size_t s = 0; s < n_scen; ++s) {
    const double* row = scenarios.data.row(s);
    double v1 = 0.0;
    for (std::size_t i = 0; i < n_legs; ++i) {
      double shift = 0.0;
      if (fx_col[i] >= 0) shift += row[fx_col[i]];
      if (ir_col[i] >= 0) shift += neg_expiry_[i] * row[ir_col[i]];
      v1 += (shift == 0.0) ? value0_[i] : value0_[i] * std::exp(shift);
    }
    const double s1_base =
        (base_col >= 0) ? s0_base_ * std::exp(row[base_col]) : s0_base_;
    out[s] = v1 / s1_base - v0_base;
  }
  return out;
}

std::vector<double> CompiledBook::linear_exposures(double bump) const {
  if (!(bump > 0.0)) throw std::invalid_argument("bump must be > 0");
  std::vector<double> w(factors_.size(), 0.0);
  std::vector<double> shocks(factors_.size(), 0.0);
  for (std::size_t j = 0; j < factors_.size(); ++j) {
    shocks[j] = bump;
    const double up = pnl(shocks.data());
    shocks[j] = -bump;
    const double dn = pnl(shocks.data());
    shocks[j] = 0.0;
    w[j] = (up - dn) / (2.0 * bump);
  }
  return w;
}

}  // namespace fxvar
