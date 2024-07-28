#include <iostream>

#include <algorithm>
#include <vector>
#include <optional>
#include <queue>
#include <stack>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <cassert>
#include <sstream>
#include <unordered_set>

#define _USE_MATH_DEFINES
#include <math.h>

#include <GL/glew.h>

#include <glfwpp/glfwpp.h>

#undef MATHTER_ENABLE_SIMD

#include <Mathter/Vector.hpp>
#include <Mathter/Matrix.hpp>

namespace math = mathter;

using float2 = math::Vector<float, 2>;
using float3 = math::Vector<float, 3>;

using float3x3 = math::Matrix<float, 3, 3>;

static constexpr int no_neighboard = -1;

struct int3 {
  int a;
  int b;
  int c;

  int &operator [] (const int index)
  {
    assert(index >= 0);
    assert(index < 3);
    return reinterpret_cast<int *>(this)[index];
  }

  const int &operator [] (const int index) const
  {
    assert(index >= 0);
    assert(index < 3);
    return reinterpret_cast<const int *>(this)[index];
  }

  int *begin()
  {
    return reinterpret_cast<int *>(this);
  }

  const int *begin() const
  {
    return reinterpret_cast<const int *>(this);
  }

  int *end()
  {
    return reinterpret_cast<int *>(this) + 3;
  }

  const int *end() const
  {
    return reinterpret_cast<const int *>(this) + 3;
  }
};

struct int2 {
  int a;
  int b;

  int &operator [] (const int index)
  {
    assert(index >= 0);
    assert(index < 2);
    return reinterpret_cast<int *>(this)[index];
  }

  const int &operator [] (const int index) const
  {
    assert(index >= 0);
    assert(index < 2);
    return reinterpret_cast<const int *>(this)[index];
  }

  int *begin()
  {
    return reinterpret_cast<int *>(this);
  }

  const int *begin() const
  {
    return reinterpret_cast<const int *>(this);
  }

  int *end()
  {
    return reinterpret_cast<int *>(this) + 2;
  }

  const int *end() const
  {
    return reinterpret_cast<const int *>(this) + 2;
  }
};

std::ostream &operator<<(std::ostream &stream, float2 vector)
{
  //  stream << "(" << vector.x << ", " << vector.y << ")";
  return stream;
}

std::ostream &operator<<(std::ostream &stream, float3 vector)
{
  //  stream << "(" << vector.x << ", " << vector.y << ", " << vector.z << ")";
  return stream;
}

std::ostream &operator<<(std::ostream &stream, int2 vector)
{
  //  stream << "(" << vector.a << ", " << vector.b << ")";
  return stream;
}

std::ostream &operator<<(std::ostream &stream, int3 vector)
{
  //  stream << "(" << vector.a << ", " << vector.b << ", " << vector.c << ")";
  return stream;
}

template<typename T>
std::ostream &operator<<(std::ostream &stream, std::vector<T> vector)
{
  int i = 0;
  //  stream << "(";
  for (const T &value : vector) {
    //  stream << i << ": " << value << (&value == &vector.back() ? "" : ", ");
    i++;
  }
  //  stream << ")";
  return stream;
}

class Geometry {
 public:
  virtual ~Geometry() = default;
  virtual std::shared_ptr<Geometry> copy() const = 0;
};

class Points : public Geometry {
 public:
  std::vector<float2> points;
  
  ~Points() override = default;

  std::shared_ptr<Geometry> copy() const override
  {
    return std::dynamic_pointer_cast<Geometry>(std::make_shared<Points>(*this));
  }
};

class Noise : public Points {
 public:
  int total = 0;
  Noise(int total_) : total(total_) {}
  ~Noise() override = default;

  std::shared_ptr<Geometry> copy() const override
  {
    return std::dynamic_pointer_cast<Geometry>(std::make_shared<Noise>(*this));
  }
};

class PointsLine : public Geometry {
 public:
  std::shared_ptr<Points> base_points;
  std::vector<int> point_line;

  ~PointsLine() override = default;

  std::shared_ptr<Geometry> copy() const override
  {
    return std::dynamic_pointer_cast<Geometry>(std::make_shared<PointsLine>(*this));
  }
};

struct UnorderedEdge {
  int a;
  int b;
  
  UnorderedEdge(int input_a, int input_b) : a(input_a), b(input_b)
  {
    if (a < b) {
      std::swap(a, b);
    }
  }
};

struct UnorderedEdgeEqual {
  bool operator() (const UnorderedEdge a_edge, const UnorderedEdge b_edge) const
  {
    return a_edge.a == b_edge.a && a_edge.b == b_edge.b;
  }
};

struct UnorderedEdgeHash {
  std::size_t operator() (const UnorderedEdge edge) const
  {
    std::size_t hash = {};
    hash |= std::size_t(edge.a);
    hash |= std::size_t(edge.b) << sizeof(edge.a) * 8;
    return hash;
  }
};

using EdgesSet = std::unordered_set<UnorderedEdge, UnorderedEdgeHash, UnorderedEdgeEqual>;

class Trangulation : public Geometry {
 public:
  std::shared_ptr<Points> base_points;
  std::vector<int3> triangles;
  EdgesSet edges;

  std::vector<float2> positions;

  ~Trangulation() override = default;

  std::shared_ptr<Geometry> copy() const override
  {
    return std::dynamic_pointer_cast<Geometry>(std::make_shared<Trangulation>(*this));
  }
};

class Data {
 public:
  std::shared_ptr<Geometry> main_geometry;

  std::vector<std::shared_ptr<Geometry>> generated_geometry;
};

static void ensure_editable_points(Data &data)
{
  if (!data.main_geometry) {
    data.main_geometry = std::make_shared<Points>();
  }
  assert(dynamic_cast<const Points *>(data.main_geometry.get()));
}

static void add_point(Data &data, const float2 point)
{
  std::vector<float2> &points = dynamic_cast<Points &>(*data.main_geometry).points;
  points.emplace(points.end(), point);
}

static float3 to_float3(const float2 input)
{
  float3 result;
  result.x = input.x;
  result.y = input.y;
  result.z = 0.0f;
  return result;
}

static std::shared_ptr<PointsLine> convex_hull(const std::shared_ptr<Points> points)
{
  std::shared_ptr<PointsLine> convex_outline = std::make_shared<PointsLine>();
  convex_outline->base_points = points;

  if (points->points.size() < 3) {
    convex_outline->point_line.resize(points->points.size());
    std::iota(convex_outline->point_line.begin(), convex_outline->point_line.end(), 0);
    return convex_outline;
  }

  const float2 mean = std::accumulate(points->points.begin(), points->points.end(), float2(0.0f)) / points->points.size();
  const int max_i = std::distance(points->points.begin(),
    std::max_element(points->points.begin(), points->points.end(), [&](const float2 a, const float2 b) {
      return a.y < b.y;
    }));

  std::vector<int> sorted_convex;
  sorted_convex.resize(points->points.size());
  std::iota(sorted_convex.begin(), sorted_convex.end(), 0);
  std::sort(sorted_convex.begin(), sorted_convex.end(), [&](const int a_i, const int b_i) {
    const float2 a_point = points->points[a_i] - points->points[max_i];
    const float2 b_point = points->points[b_i] - points->points[max_i];
    return atan2(a_point.x, a_point.y) > atan2(b_point.x, b_point.y);
  });
  
  std::vector<int> result_stack;
  for (const int point_i : sorted_convex) {
    while (true) {
      if (result_stack.size() < 2) {
        break;
      }

      const int last_stack_i = result_stack.size() - 1;
      const float3 current_point = to_float3(points->points[point_i]);
      const float3 a_point = to_float3(points->points[result_stack[last_stack_i]]) - current_point;
      const float3 b_point = current_point - to_float3(points->points[result_stack[last_stack_i - 1]]);
      const float3 tangent = math::Cross<float, 3, false>(std::array<const float3 *, 2>{&a_point, &b_point});
      if (tangent.z > 0) {
        break;
      }
      result_stack.pop_back();
    }
    result_stack.emplace(result_stack.end(), point_i);
  }
  
  convex_outline->point_line = std::move(result_stack);
  return convex_outline;
}

static bool is_index(const int index)
{
  return index >= 0;
}

static void push_tris_as_edges(const std::vector<int3> &tris, EdgesSet &edges)
{
  for (const int3 tri : tris) {
    edges.insert({tri.a, tri.b});
    edges.insert({tri.b, tri.c});
    edges.insert({tri.c, tri.a});
  }
}

static float squared(const float v)
{
  return v * v;
}

static void delaunay_system(const float2 a, const float2 b, const float2 c, float &r_a, float &r_b, float &r_c, float &r_d)
{
  r_a = math::Determinant(float3x3(a.x, a.y, 1.0f,
                                  b.x, b.y, 1.0f,
                                  c.x, c.y, 1.0f));

  r_b = math::Determinant(float3x3(squared(a.x) + squared(a.y), a.y, 1.0f,
                                   squared(b.x) + squared(b.y), b.y, 1.0f,
                                   squared(c.x) + squared(c.y), c.y, 1.0f));

  r_c = math::Determinant(float3x3(squared(a.x) + squared(a.y), a.x, 1.0f,
                                   squared(b.x) + squared(b.y), b.x, 1.0f,
                                   squared(c.x) + squared(c.y), c.x, 1.0f));

  r_d = math::Determinant(float3x3(squared(a.x) + squared(a.y), a.x, a.y,
                                   squared(b.x) + squared(b.y), b.x, b.y,
                                   squared(c.x) + squared(c.y), c.x, c.y));
}

static float2 tris_centre(const float a, const float b, const float c, const float /*d*/)
{
  return float2(b, -c) / (a * 2.0f);
}

/*
static float tris_radius(const float a, const float b, const float c, const float d)
{
  return std::sqrt((squared(b) + squared(c) - 4.0f * a * d) / (4.0f * squared(a)));
}
*/

static std::pair<float2, float> circle_info_for(const float2 point_a, const float2 point_b, const float2 point_c)
{
  float a;
  float b;
  float c;
  float d;
  delaunay_system(point_a, point_b, point_c, a, b, c, d);
  const float2 centre = tris_centre(a, b, c, d);
  return std::make_pair(centre, /*tris_radius(a, b, c, d)*/ math::Length(centre - point_a));
}

static float angle_between(const float2 ab, const float2 bc)
{
  const float dot = ab.x * bc.x + ab.y * bc.y;
  const float det = ab.x * bc.y - ab.y * bc.x;
  const float angle = atan2(det, dot);
  return angle + float(angle < 0.0f) * M_PI * 2.0f;
}

static bool is_point_in_triangle(const std::vector<float2> &positions, const int3 tri, const float2 point)
{
  const float2 ap = point - positions[tri.a];
  const float2 bp = point - positions[tri.b];
  const float2 cp = point - positions[tri.c];
  
  const float2 ab = positions[tri.b] - positions[tri.a];
  const float2 bc = positions[tri.c] - positions[tri.b];
  const float2 ca = positions[tri.a] - positions[tri.c];
  
  const float dot_a = ap.x * ab.y - ap.y * ab.x;
  const float dot_b = bp.x * bc.y - bp.y * bc.x;
  const float dot_c = cp.x * ca.y - cp.y * ca.x;
  
  return (dot_a > 0.0f) == (dot_b > 0.0f) && (dot_b > 0.0f) == (dot_c > 0.0f);
}

template<typename T>
static void delete_and_reorder(std::vector<T> &data, const int index)
{
  std::swap(data[index], data.back());
  data.resize(data.size() - 1);
}

template<typename T>
static T mod(const T v, const T s)
{
    return ((v % s) + s) % s;
}

namespace topo_set {

static bool part_of(const int3 all, const int part)
{
  return (part == all.a || part == all.b || part == all.c);
}

static bool equals(const int3 left, const int3 right)
{
  return part_of(left, right.a) && part_of(left, right.b) && part_of(left, right.b);
}

static bool part_of(const int3 all, const int2 part)
{
  return part_of(all, part.a) && part_of(all, part.b);
}

static bool part_of(const int2 all, const int part)
{
  return part == all.a || part == all.b;
}

static bool equals(const int2 left, const int2 right)
{
  return part_of(left, right.a) && part_of(left, right.b);
}

static int index_of(const int3 all, const int part)
{
  assert(part_of(all, part));
  return int(all.b == part) * 1 + int(all.c == part) * 2;
}

static int index_of(const int2 all, const int part)
{
  assert(part_of(all, part));
  return int(all.b == part) * 1;
}

static int unordered_index_of(const int3 all, const int2 part)
{
  assert(part_of(all, part));
  return int(equals(int2{all.b, all.c}, part)) * 1 + int(equals(int2{all.c, all.a}, part)) * 2;
}

static int diff(const int3 all, const int2 part)
{
  assert(part_of(all, part));
  assert(all.a >= 0 && all.b >= 0 && all.c >= 0);
  return (all.a - part.a) + (all.b - part.b) + all.c;
}

static int diff(const int2 all, const int part)
{
  assert(part_of(all, part));
  assert(all.a >= 0 && all.b >= 0);
  return (all.a - part) + all.b;
}

static int2 sample(const int3 all, const int2 indices)
{
  return int2{all[indices.a], all[indices.b]};
}

static const std::array<int2, 3> edges{int2{0, 1}, int2{1, 2}, int2{2, 0}};
static const int3 face{0, 1, 2};

static const std::array<int, 3> shift_front{1, 2, 0};
static const std::array<int, 3> shift_back{2, 0, 1};

constexpr int vert_a = 0;
constexpr int vert_b = 1;
constexpr int vert_c = 2;

constexpr int edge_ab = 0;
constexpr int edge_bc = 1;
constexpr int edge_ca = 2;

}

static bool is_valid_self_map(const std::vector<int3> &data)
{
  for (int i = 0; i < data.size(); i++) {
    const int3 elem = data[i];
    if ((elem.a == elem.b && is_index(elem.a)) ||
        (elem.b == elem.c && is_index(elem.b)) ||
        (elem.c == elem.a && is_index(elem.c))) {
      return false;
    }
    if (std::any_of(elem.begin(), elem.end(), [i](const int v) { return i == v;})) {
      /* No self connections. */
      return false;
    }
    for (const int e_i : {0, 1, 2}) {
      if (!is_index(elem[e_i])) {
        continue;
      }
      const int3 other = data[elem[e_i]];
      if (!std::any_of(other.begin(), other.end(), [i](const int v) { return i == v;})) {
        return false;
      }
    }
  }
  return true;
}

static bool is_valid_triangles(const std::vector<int3> &data)
{
  for (int i = 0; i < data.size(); i++) {
    const int3 elem = data[i];
    if (((elem.a == elem.b) && is_index(elem.a)) ||
        ((elem.b == elem.c) && is_index(elem.b)) ||
        ((elem.c == elem.a) && is_index(elem.c))) {
      return false;
    }
  }
  return true;
}

static void build_polygon(const std::vector<std::pair<float2, float>> &tris_params,
                          const std::vector<int3> &tris_verts,
                          const std::vector<int3> &tris_neighboards,
                          const int start_tri_i,
                          const int start_side_i,
                          const float2 point_to_insert,
                          std::vector<int> &r_polygon_soup,
                          std::vector<int> &r_polygon_verts,
                          std::vector<int> &r_polygon_neighboards,
                          std::vector<int> &r_polygon_to_remap)
{
  const int next_tri_i = tris_neighboards[start_tri_i][start_side_i];
  if (!is_index(next_tri_i)) {
    r_polygon_verts.insert(r_polygon_verts.end(), tris_verts[start_tri_i][start_side_i]);
    r_polygon_neighboards.insert(r_polygon_neighboards.end(), no_neighboard);
    r_polygon_to_remap.insert(r_polygon_to_remap.end(), no_neighboard);
    return;
  }

  const auto [centre, radius] = tris_params[next_tri_i];
  if (math::Length(centre - point_to_insert) >= radius) {
    r_polygon_verts.insert(r_polygon_verts.end(), tris_verts[start_tri_i][start_side_i]);
    r_polygon_neighboards.insert(r_polygon_neighboards.end(), next_tri_i);
    r_polygon_to_remap.insert(r_polygon_to_remap.end(), start_tri_i);
    return;
  }

  assert(std::find(r_polygon_soup.begin(), r_polygon_soup.end(), next_tri_i) == r_polygon_soup.end());
  r_polygon_soup.insert(r_polygon_soup.end(), next_tri_i);

  const int3 next_verts = tris_verts[next_tri_i];
  const int2 side_edge = topo_set::sample(tris_verts[start_tri_i], topo_set::edges[start_side_i]);
  const int other_side_i = topo_set::unordered_index_of(next_verts, side_edge);

  build_polygon(tris_params,
                tris_verts,
                tris_neighboards,
                next_tri_i,
                topo_set::shift_front[other_side_i],
                point_to_insert,
                r_polygon_soup,
                r_polygon_verts,
                r_polygon_neighboards,
                r_polygon_to_remap);
  build_polygon(tris_params,
                tris_verts,
                tris_neighboards,
                next_tri_i,
                topo_set::shift_back[other_side_i],
                point_to_insert,
                r_polygon_soup,
                r_polygon_verts,
                r_polygon_neighboards,
                r_polygon_to_remap);
}

#ifndef NDEBUG
  #define assert_msg(str_name, condition)\
    if (!(condition)) {\
      std::cout << str_name.str();\
    }\
    assert(condition);
#else
  #define assert_msg(str_name, condition)
#endif

static std::shared_ptr<Trangulation> delaunay(const std::shared_ptr<Points> points)
{
  std::stringstream stream;

  std::shared_ptr<Trangulation> mesh = std::make_shared<Trangulation>();
  mesh->base_points = points;

  if (points->points.size() < 3) {
    return mesh;
  }

  if (points->points.size() == 3) {
    mesh->triangles = {{0, 1, 2}};
    push_tris_as_edges(mesh->triangles, mesh->edges);
    mesh->positions = points->points;
    return mesh;
  }

  const int total_points = points->points.size();

  const int3 super_tri{total_points + 0, total_points + 1, total_points + 2};
  std::vector<int3> tris_verts = {super_tri};
  tris_verts.reserve(total_points);

  std::vector<float2> positions;
  positions.resize(total_points + 3);
  std::copy(points->points.begin(), points->points.end(), positions.begin());
  positions[super_tri.a] = float2(1000.0f, 1000.0f);
  positions[super_tri.b] = float2(-1000.0f, 1000.0f);
  positions[super_tri.c] = float2(0.0f, -1000.0f);

  // positions[super_tri.a] = float2(0.01f, 0.01f);
  // positions[super_tri.b] = float2(0.99f, 0.01f);
  // positions[super_tri.c] = float2(0.01f, 0.99f);

  std::vector<int3> tris_neighboards = {{no_neighboard, no_neighboard, no_neighboard}};
  tris_neighboards.reserve(total_points);

  std::vector<std::vector<int>> tris_verts_to_insert = {{}};
  tris_verts_to_insert.front().resize(total_points);
  std::iota(tris_verts_to_insert.front().begin(), tris_verts_to_insert.front().end(), 0);

  assert_msg(stream, std::all_of(tris_verts_to_insert.front().begin(),
                                 tris_verts_to_insert.front().end(),
                                 [&](const int vert_i){
                                   return is_point_in_triangle(positions, super_tri, positions[vert_i]);
                                 }));

  std::vector<std::pair<float2, float>> tris_params = {std::make_pair(float2(0.0f, 0.0f), 10000.0f)};
  tris_params.reserve(total_points);

  std::vector<int> polygon_soup;
  std::vector<int> polygon_verts;
  std::vector<int> polygon_neighboards;
  std::vector<int> polygon_to_remap;

  std::vector<int> points_soup;

  std::queue<int> tris_to_split;
  tris_to_split.push(0);
  while (!tris_to_split.empty()) {
    assert_msg(stream, tris_verts.size() == tris_neighboards.size());
    assert_msg(stream, tris_verts.size() == tris_verts_to_insert.size());
    assert_msg(stream, tris_verts.size() == tris_params.size());
    const int first_tris_to_split = tris_to_split.front();
    tris_to_split.pop();

    if (tris_verts_to_insert[first_tris_to_split].empty()) {
      continue;
    }

    assert_msg(stream, is_valid_self_map(tris_neighboards));
    assert_msg(stream, is_valid_triangles(tris_verts));

    //  stream << "Split tri:\t" << first_tris_to_split << ";\n";

    const int3 verts = tris_verts[first_tris_to_split];

    //  stream << "  verts:\t" << verts << ";\n";
    //  stream << "  tris_neighboards[first_tris_to_split]:\t" << tris_neighboards[first_tris_to_split] << ";\n";

    assert_msg(stream, std::all_of(tris_verts_to_insert[first_tris_to_split].begin(),
                                   tris_verts_to_insert[first_tris_to_split].end(),
                                   [&](const int vert_i){
                                     return is_point_in_triangle(positions, verts, positions[vert_i]);
                                   }));

    const int insert_vert = tris_verts_to_insert[first_tris_to_split][0];
    delete_and_reorder(tris_verts_to_insert[first_tris_to_split], 0);

    //  stream << "  insert_vert:\t" << insert_vert << ";\n";
    //  stream << "  positions[insert_vert]:\t" << positions[insert_vert] << ";\n";

    const float2 point_to_insert = positions[insert_vert];

    polygon_soup.clear();
    polygon_verts.clear();
    polygon_neighboards.clear();
    polygon_to_remap.clear();

    polygon_soup.insert(polygon_soup.end(), first_tris_to_split);

    for (const int corner_i : topo_set::face) {
      build_polygon(tris_params,
                    tris_verts,
                    tris_neighboards,
                    first_tris_to_split,
                    corner_i,
                    point_to_insert,
                    polygon_soup,
                    polygon_verts,
                    polygon_neighboards,
                    polygon_to_remap);
    }

    assert_msg(stream, polygon_verts.size() == polygon_neighboards.size());
    assert_msg(stream, polygon_verts.size() == polygon_to_remap.size());
    const int new_tris_num = polygon_verts.size();

    //  stream << "  Polygon soup:" << ";\n";
    //  stream << "    Polygon polygon_soup:\t" << polygon_soup << ";\n";
    //  stream << "    Polygon polygon_verts:\t" << polygon_verts << ";\n";
    //  stream << "    Polygon polygon_neighboards:\t" << polygon_neighboards << ";\n";
    //  stream << "    Polygon polygon_to_remap:\t" << polygon_to_remap << ";\n";

    points_soup.clear();
    points_soup.reserve(std::accumulate(polygon_soup.begin(), polygon_soup.end(), 0,
        [&](const int value, const int tri_i) -> int { return value + tris_verts_to_insert[tri_i].size(); }));
    for (const int tri_i : polygon_soup) {
      points_soup.insert(points_soup.end(), tris_verts_to_insert[tri_i].begin(), tris_verts_to_insert[tri_i].end());
    }

    /* Polar sort of points around inserted vertex to split them next to polar segments by inserted edges. */
    const float2 polygon_point = positions[verts.a];
    const float2 insert_edge = polygon_point - point_to_insert;

    const auto inserting_angle = [&](const int i) -> float {
      return angle_between(insert_edge, positions[i] - point_to_insert); };
    std::sort(points_soup.begin(), points_soup.end(), [&](const int a_i, const int b_i) {
      return inserting_angle(a_i) < inserting_angle(b_i);
    });

    const int size_difference = new_tris_num - polygon_soup.size();
    assert_msg(stream, size_difference >= 0);
    if (size_difference > 0) {
      //  stream << "  Before extend:" << ";\n";
      //  stream << "    Extended tris_verts_to_insert:\t" << polygon_soup << ";\n";
      //  stream << "    Extended tris_verts:\t" << polygon_verts << ";\n";
      //  stream << "    Extended tris_neighboards:\t" << polygon_neighboards << ";\n";
      //  stream << "    Extended points_soup:\t" << polygon_to_remap << ";\n";

      const int start_of_extra = tris_verts_to_insert.size();
      tris_verts_to_insert.resize(start_of_extra + size_difference);
      tris_verts.resize(start_of_extra + size_difference);
      tris_neighboards.resize(start_of_extra + size_difference);
      tris_params.resize(start_of_extra + size_difference);

      polygon_soup.resize(polygon_verts.size());
      std::iota(polygon_soup.end() - size_difference, polygon_soup.end(), start_of_extra);

      //  stream << "  After extend:" << ";\n";
      //  stream << "    Extended polygon_soup:\t" << polygon_soup << ";\n";
      //  stream << "    Extended polygon_verts:\t" << polygon_verts << ";\n";
      //  stream << "    Extended polygon_neighboards:\t" << polygon_neighboards << ";\n";
      //  stream << "    Extended polygon_to_remap:\t" << polygon_to_remap << ";\n";
    }

    //  stream << "  Before aprtial remap:" << ";\n";
    //  stream << "    tris_neighboards:\t" << tris_neighboards << ";\n";

    for (int index = 0; index < new_tris_num; index++) {
      const int connection = polygon_neighboards[index];
      const int old_tri_i = polygon_to_remap[index];
      const int new_tri_i = polygon_soup[index];
      if (is_index(connection) && is_index(old_tri_i)) {
        const int swap_dummy_index = std::numeric_limits<int>::min() + new_tri_i;
        assert_msg(stream, connection >= 0);
        assert_msg(stream, connection < tris_neighboards.size());
        std::replace(tris_neighboards[connection].begin(), tris_neighboards[connection].end(), old_tri_i, swap_dummy_index);
      }
    }
    for (int index = 0; index < new_tris_num; index++) {
      const int connection = polygon_neighboards[index];
      const int old_tri_i = polygon_to_remap[index];
      const int new_tri_i = polygon_soup[index];
      if (is_index(connection) && is_index(old_tri_i)) {
        const int swap_dummy_index = std::numeric_limits<int>::min() + new_tri_i;
        assert_msg(stream, connection >= 0);
        assert_msg(stream, connection < tris_neighboards.size());
        std::replace(tris_neighboards[connection].begin(), tris_neighboards[connection].end(), swap_dummy_index, new_tri_i);
      }
    }

    //  stream << "  After aprtial remap:" << ";\n";
    //  stream << "    tris_neighboards:\t" << tris_neighboards << ";\n";

    for (int index = 0; index < new_tris_num; index++) {
      const int new_tri_i = polygon_soup[index];

      const int prev_index = mod<int>(index - 1, new_tris_num);
      const int next_index = mod<int>(index + 1, new_tris_num);
      const int vert_a = polygon_verts[index];
      const int vert_b = polygon_verts[next_index];
      const int3 sub_face_verts{vert_a, vert_b, insert_vert};
      tris_verts[new_tri_i] = sub_face_verts;

      const int3 sub_face_neighboards{polygon_neighboards[index], polygon_soup[next_index], polygon_soup[prev_index]};
      tris_neighboards[new_tri_i] = sub_face_neighboards;

      const float2 left_edge = positions[vert_a] - point_to_insert;
      const float2 right_edge = positions[vert_b] - point_to_insert;

      std::vector<int> sub_tri_verts_to_insert;
      const auto begin = std::lower_bound(points_soup.begin(), points_soup.end(), inserting_angle(vert_a), [&](const int point_i, const float angle) {
        return inserting_angle(point_i) < angle;
      });
      const auto end = std::upper_bound(points_soup.begin(), points_soup.end(), angle_between(left_edge, right_edge), [&](const float angle, const int point_i) {
        return angle < inserting_angle(point_i) - inserting_angle(vert_a);
      });
      sub_tri_verts_to_insert.insert(sub_tri_verts_to_insert.end(), begin, end);
      //  stream << "    sub_tri_verts_to_insert:\t" << sub_tri_verts_to_insert << ";\n";
      tris_verts_to_insert[new_tri_i] = std::move(sub_tri_verts_to_insert);

      if (!tris_verts_to_insert[new_tri_i].empty()) {
        tris_to_split.push(new_tri_i);
      }

      tris_params[new_tri_i] = circle_info_for(positions[sub_face_verts.a], positions[sub_face_verts.b], positions[sub_face_verts.c]);

      //  stream << "      tris_verts_to_insert:\t" << tris_verts_to_insert << ";\n";
      //  stream << "      tris_verts:\t" << tris_verts << ";\n";
      //  stream << "      tris_neighboards:\t" << tris_neighboards << ";\n";
    }

    //  stream << "  After all:" << ";\n";
    //  stream << "    tris_verts_to_insert:\t" << tris_verts_to_insert << ";\n";
    //  stream << "    tris_verts:\t" << tris_verts << ";\n";
    //  stream << "    tris_neighboards:\t" << tris_neighboards << ";\n";
  }

  assert_msg(stream, std::all_of(tris_verts_to_insert.begin(), tris_verts_to_insert.end(), [](const std::vector<int> &verts) { return verts.empty(); }));

  // std::cout << stream.str();

  // std::cout << tris_verts << ";\n";
  // std::cout << "\n";
  // std::cout << "\n";

  std::vector<int3> result_tris;
  result_tris.reserve(tris_verts.size());
  std::copy_if(tris_verts.begin(), tris_verts.end(), std::back_inserter(result_tris), [&](const int3 tri) {
    return !(topo_set::part_of(super_tri, tri.a) || topo_set::part_of(super_tri, tri.b) || topo_set::part_of(super_tri, tri.c));
  });

  mesh->triangles = std::move(result_tris);
  push_tris_as_edges(mesh->triangles, mesh->edges);
  // mesh->positions = std::move(positions);

  return mesh;
}

static void ensure_noise_doamin(Noise &noise)
{
  const int diff = noise.total - noise.points.size();
  noise.points.resize(noise.total);
  if (diff > 0) {
    std::generate(noise.points.end() - diff, noise.points.end(), []() -> float2 {
      const float x = float(rand()) / float(RAND_MAX);
      const float y = float(rand()) / float(RAND_MAX);
      return float2{x, y};
    });
  }
}

static void update_generated_geometry(Data &data)
{
  std::shared_ptr<Points> points = std::dynamic_pointer_cast<Points>(data.main_geometry);
  if (std::shared_ptr<Noise> noise = std::dynamic_pointer_cast<Noise>(data.main_geometry)) {
    ensure_noise_doamin(*noise);
  }

  data.generated_geometry.clear();

  std::shared_ptr<Points> base_points = std::dynamic_pointer_cast<Points>(data.main_geometry);
  if (!base_points) {
    return;
  }

  // data.generated_geometry.emplace(data.generated_geometry.end(), convex_hull(base_points));
  data.generated_geometry.emplace(data.generated_geometry.end(), delaunay(base_points));
}

struct Context {
  float2 size;
  float point_size;
};

static void draw_points(const std::vector<float2> &points)
{
  glColor4f(1, 0, 0, 1);
  glBegin(GL_POINTS);
  for (const float2 point : points) {
    glVertex3f(point.x, point.y, 0);
  }
  glEnd();
}

static void draw_line(const std::vector<float2> &points, const std::vector<int> &indices)
{
  glBegin(GL_LINE_LOOP);
  for (const int point_i : indices) {
    const float2 point = points[point_i];
    glVertex3f(point.x, point.y, 0);
  }
  glEnd();
}

static void draw_edges(const std::vector<float2> &points, const EdgesSet &edges)
{
  glColor4f(1, 0, 1, 1);
  glBegin(GL_LINES);
  for (const UnorderedEdge edge : edges) {
    const float2 a_point = points[edge.a];
    const float2 b_point = points[edge.b];
    glVertex3f(a_point.x, a_point.y, 0);
    glVertex3f(b_point.x, b_point.y, 0);
  }
  glEnd();
}

static void draw_circle(const float2 centre, const float radius, const int details)
{
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < details; i++) {
    const float angle = float(i) / details * M_PI * 2.0f;
    const float2 pos = centre + float2(std::sin(angle), std::cos(angle)) * radius;
    glVertex3f(pos.x, pos.y, 0.0f);
  }
  glEnd();
}

static void draw_tri_circles(const std::vector<float2> &points, const std::vector<int3> &tris)
{
  const float hue_step = 0.2f;
  float hue_iter = 0.0f;
  for (const int3 tri : tris) {
    hue_iter += hue_step;
    glColor4f((sin(hue_iter) + 1.0) / 2.0, (cos(hue_iter) + 1.0) / 2.0, (-sin(hue_iter) + 1.0) / 2.0, 1.0);
    const auto [centre, radius] = circle_info_for(points[tri.a], points[tri.b], points[tri.c]);
    draw_circle(centre, radius, 50);
  }
}

static void draw_tris(const std::vector<float2> &points, const std::vector<int3> &tris, const float offset)
{
  glColor4f(1, 1, 0, 1);
  glBegin(GL_TRIANGLES);
  for (const int3 tri : tris) {
    const float2 ab = math::SafeNormalize(points[tri.a] - points[tri.b]);
    const float2 bc = math::SafeNormalize(points[tri.b] - points[tri.c]);
    const float2 ca = math::SafeNormalize(points[tri.c] - points[tri.a]);

    const float2 bac_line = math::SafeNormalize(ab - ca) * offset;
    const float2 abc_line = math::SafeNormalize(bc - ab) * offset;
    const float2 bca_line = math::SafeNormalize(ca - bc) * offset;

    const float2 a = points[tri.a] - bac_line;
    const float2 b = points[tri.b] - abc_line;
    const float2 c = points[tri.c] - bca_line;

    glVertex3f(a.x, a.y, 0.0f);
    glVertex3f(b.x, b.y, 0.0f);
    glVertex3f(c.x, c.y, 0.0f);
  }
  glEnd();
}

static void draw(const Context &draw_context, const Data &data)
{
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glOrtho(0.0f, 1.0f, 1.0f, 0.0f, -100.0f, 100.0f);

  glPointSize(draw_context.point_size);

  glColor4f(1, 1, 1, 1);

  if (data.main_geometry && dynamic_cast<const Points *>(data.main_geometry.get())) {
    draw_points(dynamic_cast<Points &>(*data.main_geometry).points);
  }

  for (const std::shared_ptr<Geometry> &geometry : data.generated_geometry) {
    if (const PointsLine *line = dynamic_cast<const PointsLine *>(geometry.get())) {
      const Points &points_data = *line->base_points;
      draw_line(points_data.points, line->point_line);
    } else if (const Trangulation *mesh = dynamic_cast<const Trangulation *>(geometry.get())) {
      const Points &points_data = *mesh->base_points;
      draw_edges(points_data.points /*mesh->positions*/, mesh->edges);
      draw_tris(points_data.points, mesh->triangles, 0.01f);
      draw_tri_circles(points_data.points, mesh->triangles);
      draw_points(points_data.points);
    }
  }
}

int main()
{
  [[maybe_unused]] const auto GLFW = glfw::init();

  glfw::Window window{640, 480, "Delaunay Triangulation"};

  std::vector<Data> history = {{}};
  std::stack<Data> undo_history;

  // history.insert(history.end(), {std::make_shared<Noise>(1000)});
  // update_generated_geometry(history.back());

  const auto point_insert_action = [&](glfw::Window &window, const float2 point) {
    Data new_sep;
    if (history.back().main_geometry) {
      new_sep.main_geometry = history.back().main_geometry->copy();
    }

    const std::tuple<int, int> window_size = window.getSize();
    const float2 view_size(std::get<0>(window_size), std::get<1>(window_size));

    ensure_editable_points(new_sep);
    add_point(new_sep, point / view_size);
    update_generated_geometry(new_sep);
    history.insert(history.end(), std::move(new_sep));
    undo_history = {};
  };

  window.mouseButtonEvent.setCallback([&](glfw::Window &window,
                                        [[maybe_unused]] const glfw::MouseButton button,
                                        const glfw::MouseButtonState state,
                                        [[maybe_unused]] const glfw::ModifierKeyBit key) {
    if (state != glfw::MouseButtonState::Press) {
      return;
    }
    const std::tuple<double, double> mouse = window.getCursorPos();
    point_insert_action(window, float2(std::get<0>(mouse), std::get<1>(mouse)));
  });

  window.keyEvent.setCallback([&](glfw::Window &window,
                                 const glfw::KeyCode key,
                                 const int code,
                                 const glfw::KeyState state,
                                 const glfw::ModifierKeyBit key_bit) {
    if (state != glfw::KeyState::Press && state != glfw::KeyState::Repeat) {
      return;
    }
    switch (key) {
      case glfw::KeyCode::X:
        if (history.size() > 1) {
          undo_history.push(std::move(history.back()));
          history.pop_back();
        }
        break;
      case glfw::KeyCode::Z:
        if (!undo_history.empty()) {
          history.insert(history.end(), std::move(undo_history.top()));
          undo_history.pop();
        }
        break;
      default:
        break;
    }
  });

  window.framebufferSizeEvent.setCallback([]([[maybe_unused]] glfw::Window &window,
                                          const int width,
                                          const int height) {
    glViewport(0, 0, width, height);
  });

  glfw::makeContextCurrent(window);
  while (!window.shouldClose()) {
    const std::tuple<int, int> current_size = window.getSize();
    Context draw_context;
    draw_context.size = float2(std::get<0>(current_size), std::get<1>(current_size));
    draw_context.point_size = 5.0f;


    double time = glfw::getTime();
    glClearColor((sin(time) + 1.0) / 6.0, (cos(time) + 1.0) / 5.0, (-sin(time) + 1.0) / 7.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    const Data &data = history.back();
    draw(draw_context, data);

    glfw::pollEvents();
    window.swapBuffers();
  }

  return 0;
}