//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

const float EPSILON = 0.000001;

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;
float lastTime = -1.0f;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, const char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
		exit(0);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
		exit(0);
	}
}

void SetUniform(unsigned int shaderProg, const char *name, float value) {
	int loc = glGetUniformLocation(shaderProg, name);
	glUniform1f(loc, value);
}

// vertex shader in GLSL
const char * vertexSource = R"(#version 330
precision highp float;

uniform mat4 M, VP, Minv;
uniform vec3 wEye;
uniform vec4 wLiPos;

layout(location = 0) in vec3 vtxPos;	// Attrib Array 0
layout(location = 1) in vec3 vtxNorm;	    // Attrib Array 1
out vec3 color;
out vec3 wNormal;
out vec3 wView;
out vec3 wLight;

void main() {
	vec4 wPos = vec4(vtxPos, 1) * M;
	gl_Position = wPos * VP; 		// transform to clipping space
	wLight = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
	wView = wEye * wPos.w - wPos.xyz;
	wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
	color = vec3(1, 1, 1);
}
)";
// fragment shader in GLSL
const char * fragmentSource = R"(#version 330
precision highp float;

uniform vec4 La, Le;
uniform vec3 kd, ks, ka;
uniform float shine;

in vec3 color;
in vec3 wNormal;
in vec3 wView;
in vec3 wLight;
out vec4 fragmentColor;

void main() {
	vec3 N = normalize(wNormal);
	vec3 V = normalize(wView);
	vec3 L = normalize(wLight);
	vec3 H = normalize(L + V);
	float cost = max(dot(N, L), 0),
	      cosd = max(dot(N, H), 0);

	vec3 color = ka * La.xyz
		+ ( kd * cost
		  + ks * pow(cosd, shine))
		  * Le.xyz;

	fragmentColor = vec4(color, 1); // extend RGB to RGBA
}
)";


struct vec3 {
	float x, y, z;

	constexpr explicit vec3(float x = 0, float y = 0, float z = 0)
		:x(x), y(y), z(z)
	{}

	constexpr vec3(const vec3 &v)
		:x(v.x), y(v.y), z(v.z)
	{}

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
	}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
	}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
	}
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
	vec3 normalize() const {
		return (*this) * (1 / (Length() + EPSILON));
	}

	vec3 getNormalized() const {
		return vec3(*this).normalize();
	}

	float Length() const { return sqrtf(x * x + y * y + z * z); }

	operator float*() { return &x; }

	void SetUniform(unsigned int shaderProg, const char *name) const {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniform3fv(loc, 1, &x);
	}

	vec3 &operator*=(float f) {
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}

	friend vec3 operator*(float f, const vec3 &v) {
		return v * f;
	}
};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

struct mat4 {
	float m[4][4];
public:
	constexpr mat4()
		:mat4(1.0f, 0.0f, 0.0f, 0.0f,
		      0.0f, 1.0f, 0.0f, 0.0f,
		      0.0f, 0.0f, 1.0f, 0.0f,
		      0.0f, 0.0f, 0.0f, 1.0f)
	{ }

	constexpr mat4(float m00, float m01, float m02, float m03,
	               float m10, float m11, float m12, float m13,
	               float m20, float m21, float m22, float m23,
	               float m30, float m31, float m32, float m33) 
		:m { { m00, m01, m02, m03 },
		     { m10, m11, m12, m13 },
		     { m20, m21, m22, m23 },
		     { m30, m31, m32, m33 } }
	{ }

	mat4(const mat4 &m) :mat4(m[0][0], m[0][1], m[0][2], m[0][3],
	                          m[1][0], m[1][1], m[1][2], m[1][3],
				  m[2][0], m[2][1], m[2][2], m[2][3],
				  m[3][0], m[3][1], m[3][2], m[3][3])
	{ }

	mat4 operator*(const mat4& right) const {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	operator float*() { return &m[0][0]; }
	operator const float*() const { return &m[0][0]; }

	float *operator[](unsigned int i) { return &m[i][0]; }
	const float *operator[](unsigned int i) const { return &m[i][0]; }

	void SetUniform(unsigned int shaderProg, const char *name) const {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniformMatrix4fv(loc, 1, GL_TRUE, reinterpret_cast<const float *>(m));
	}

	static mat4 Translate(const vec3 &v) { return Translate(v.x, v.y, v.z); }
	static mat4 Translate(float tx, float ty, float tz) {
		return mat4( 1, 0, 0, 0,
		             0, 1, 0, 0,
			     0, 0, 1, 0,
			    tx,ty,tz, 1);
	}

	static mat4 Rotate(const vec3 &v) { return Rotate(v.x, v.y, v.z); }
	static mat4 Rotate(float rx, float ry, float rz) {
		float cosrx = cosf(rx), sinrx = sinf(rx), msinrx = -sinrx,
		      cosry = cosf(ry), sinry = sinf(ry), msinry = -sinry,
		      cosrz = cosf(rz), sinrz = sinf(rz), msinrz = -sinrz;
		return mat4(1,     0,      0, 0,
		            0, cosrx, msinrx, 0,
			    0, sinrx,  cosrx, 0,
			    0,     0,      0, 1) *
		       mat4( cosry, 0, sinry, 0,
		                 0, 1,     0, 0,
		            msinry, 0, cosry, 0,
		                 0, 0,     0, 1) *
		       mat4(cosrz, msinrz, 0, 0,
		            sinrz,  cosrz, 0, 0,
			        0,      0, 1, 0,
				0,      0, 0, 1);
	}

};

// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	constexpr explicit vec4(float x = 0, float y = 0, float z = 0, float w = 1) 
		:v { x, y, z, w }
	{ }

	constexpr vec4(const vec4 &vec) 
		:v { vec.v[0], vec.v[1], vec.v[2], vec.v[3] }
	{ }

	constexpr vec4(const vec3 &v)
		:vec4(v.x, v.y, v.z, 1)
	{ }

	vec4 operator*(const mat4& mat) const {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	float operator[](size_t index) const {
		return v[index];
	}

	float &operator[](size_t index) {
		return v[index];
	}

	vec4 &operator+=(const vec4 &w) {
		for(int i = 0; i < 4; ++i) {
			v[i] += w[i];
		}
		return *this;
	}

	vec4 operator+(const vec4 &w) const {
		return (vec4(*this) += w);
	}

	vec4 &operator*=(float f) {
		for(int i = 0; i < 4; ++i) v[i] *= f;
		return *this;
	}

	vec4 operator*(float f) const {
		return (vec4(*this) *= f);
	}

	friend vec4 operator*(float f, const vec4 &v) {
		return v * f;
	}

	vec4 &operator-=(const vec4 &w) {
		return *this += -1.0f * w;
	}

	vec4 operator-(const vec4 &w) const {
		return vec4(*this) -= w;
	}

	vec4 operator-() const {
		return -1.0f * *this;
	}

	vec4 &operator/=(float f) {
		if(f == 1) return *this;
		return *this *= (1.0f / f);
	}

	vec4 operator/(float f) const {
		if(f == 1) return *this;
		return vec4(*this) /= f;
	}

	vec4 &normalize() {
		if(fabs(v[3] - 1.0f) > EPSILON) {
			for(int i = 0; i < 3; ++i) {
				v[i] /= v[3];
			}
			v[3] = 1.0f;
		}

		float len = sqrt(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2));

		for(int i = 0; i < 3; ++i) {
			v[i] /= len;
		}
		return *this;
	}

	vec4 getNormalized() const {
		return vec4(*this).normalize();
	}


	void SetUniform(unsigned int shaderProg, const char *name) const {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniform4fv(loc, 1, v);
	}

	operator vec3() const {
		return vec3(v[0], v[1], v[2]);
	}

	friend vec4 operator*(const mat4 &m, const vec4 &v) {
		vec4 result(0, 0, 0, 0);
		for(int i = 0; i < 4; ++i) {
			for(int j = 0; j < 4; ++j) {
				result[i] = v[j] * m[i][j];
			}
		}
		return result;
	}
};

// handle of the shader program
unsigned int shaderProgram;

// 3D camera
struct Camera {
	vec3  wEye, wLookat, wVup;
	float fov, asp, fp, bp;

	Camera() 
		:wEye(),
		wLookat(0, 0, 0),
		wVup(0, 0, 1),
		fov(M_PI / 3.0f), asp(1.0f), fp(1.0f), bp(100.0f)
	{
		Animate(0.0f);
	}

	mat4 V() { // view matrix
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);
		return mat4::Translate(-wEye.x, -wEye.y, -wEye.z) * 
			mat4(	u.x,  v.x,  w.x,  0.0f,
					u.y,  v.y,  w.y,  0.0f,
					u.z,  v.z,  w.z,  0.0f,
					0.0f, 0.0f, 0.0f, 1.0f );
	}
	mat4 P() { // projection matrix
		float sy = 1/tan(fov/2);
		return mat4(sy/asp, 0.0f,  0.0f,               0.0f,
				0.0f,   sy,    0.0f,               0.0f,
				0.0f,   0.0f, -(fp+bp)/(bp - fp), -1.0f,
				0.0f,   0.0f, -2*fp*bp/(bp - fp),  0.0f);
	}

	void SetVPUniform() {
		SetVPUniform(shaderProgram);
	}

	void SetVPUniform(unsigned int shaderProg, const char *name = "VP") {
		(V() * P()).SetUniform(shaderProg, name);
	}

	void SetWEyeUniform() {
		SetWEyeUniform(shaderProgram);
	}

	void SetWEyeUniform(unsigned int shaderProg, const char *name = "wEye") {
		wEye.SetUniform(shaderProg, name);
	}

	void Animate(float) {
		wEye = vec3(
				vec4(-10.0f, 0.0f, 4.0f, 1.0f)
				//* mat4::Rotate(0.0f, 0.0f, t)
				* mat4::Translate(wLookat)
			   );
	}
};

// 3D camera
Camera camera;

struct Material {
	vec3 kd, ks, ka;
	float shine;

	constexpr Material() 
		:kd(0.3, 0.3, 0.3), 
		ks(0.5, 0.5, 0.5), 
		ka(0.15, 0.15, 0.15),
		shine(2.5) 
	{ }
	constexpr Material(vec3 kd, vec3 ks, vec3 ka, float shine)
		:kd(kd), ks(ks), ka(ka), shine(shine) { }

	void SetUniforms(unsigned int shaderProg, const char *kdname = "kd", const char *ksname = "ks", const char *kaname = "ka", const char *shinename = "shine") const
	{
		kd.SetUniform(shaderProg, kdname);
		ks.SetUniform(shaderProg, ksname);
		ka.SetUniform(shaderProg, kaname);
		SetUniform(shaderProg, shinename, shine);
	}
};

struct Geometry {
	unsigned int vao, nVtx;
	Material material;

	Geometry() {
		glGenVertexArrays(1, &vao); 
		glBindVertexArray(vao);
	}

	virtual mat4 GetModelMatrix() const {
		return mat4();
	}

	virtual mat4 GetModelInverseMatrix() const {
		return mat4();
	}

	void Draw() const {
		glBindVertexArray(vao); 
		glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}

	virtual void SetUniforms(unsigned int shaderProg) {
		material.SetUniforms(shaderProg);
	}

	virtual void Animate(float) { }

	virtual ~Geometry() noexcept { }
};

struct VertexData {
	vec3 position, normal;
	float u, v;
};

class ParamSurface : public Geometry {
protected:
	VertexData *vtxData = nullptr;

public:
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N, int M) {
		if(vtxData) delete[] vtxData;

		nVtx = N * M * 6;   
		unsigned int vbo;
		glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);

		VertexData *vtxData = new VertexData[nVtx], *pVtx = vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				*pVtx++ = GenVertexData((float)i / N, (float)j / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
				*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
				*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtx * sizeof(VertexData), vtxData, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
		glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
		glEnableVertexAttribArray(2);  // AttribArray 2 = UV
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position)); 
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, u));
	}

	virtual ~ParamSurface() noexcept {
		if(vtxData) delete[] vtxData;
	}
};


struct Sphere : ParamSurface {
	vec3 center;
	float radius;

	float tz;
	float tx;

	bool moving = false;

	Sphere(vec3 c, float r) 
		:center(c), radius(r), tz(0), tx(-50)
	{
		Create(32, 16);
		material.ka.x = 0.2;
		material.ka.y = 0.2;
		material.ka.z = 0.5;
		material.kd.y = 0.8;
		material.shine = 10;
	}

	mat4 GetModelMatrix() const {
		return mat4::Translate(GetPosition());
	}

	mat4 GetModelInverseMatrix() const {
		return mat4::Translate(-1 * GetPosition());
	}

	vec3 GetPosition() const {
		return center + vec3(tx, 0, tz);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(cos(u*2*M_PI) * sin(v*M_PI),
				sin(u*2*M_PI) * sin(v*M_PI),
				cos(v*M_PI));
		vd.position = vd.normal * radius;
		vd.u = u; vd.v = v;
		return vd;
	}

	void Animate(float t) {
		if(!moving) return;

		tx += (t - lastTime) * 9.0f;
	}

	bool intersect(const Sphere &s) const {
		vec3 c1 = s.center + vec3(s.tx, 0, s.tz);
		vec3 c2 = center + vec3(tx, 0, tz);
		vec3 dist = c2 - c1;
		float len = dist.Length();
		return len < (radius + s.radius);
	}

	void setCollided() {
		material.ka = vec3(1, 0, 0);
		material.ks = vec3(1, 0, 0);
		material.kd = vec3(1, 0, 0);
	}
};

struct Desert : ParamSurface {
	float w, h;
	
	Desert(float width, float height)
		:w(width), h(height) 
	{ 
		Create(2, 2);

		material.ka = vec3(0.2, 0.2, 0.0);
		material.kd = vec3(0.7, 0.7, 0);
		material.ks = vec3(0.1, 0.1, 0.1);
		material.shine = 0.2;
	}


	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(0.0f, 0.0f, 1.0f);
		vd.position = vec3(-w / 2 + u * w, -h / 2 + v * h, -0.001f);
		vd.u = u; vd.v = v;
		return vd;
	}
};

struct Footbridge : ParamSurface {
	float w, h;

	Footbridge(float width, float height)
		:w(width), h(height)
	{
		Create(2, 2);

		material.ka = vec3(0.6, 0.4 ,0.2);
		material.kd = vec3(0.6, 0.4, 0.2);
		material.ks = vec3();
		material.shine = 0.1;
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(0.0f, 0.0f, 1.0f);
		vd.position = vec3(-w / 2 + u * w, -h / 2 + v * h, 0);
		vd.u = u; vd.v = v;
		return vd;
	}
};


struct LightSource {
	vec4 pos;
	vec4 ambientColor;
	vec4 pointColor;

	LightSource() { }
	LightSource(vec4 pos, vec4 ambientColor, vec4 pointColor)
		:pos(pos), ambientColor(ambientColor), pointColor(pointColor) { }

	void SetUniforms(unsigned int shaderProg, const char *posname = "wLiPos", const char *ambientName = "La", const char *pointName = "Le") {
		pos.SetUniform(shaderProg, posname);
		ambientColor.SetUniform(shaderProg, ambientName);
		pointColor.SetUniform(shaderProg, pointName);
	}
};

struct Snake : ParamSurface {
	float t;

	Snake() {
		Animate(0.0f);
	}

	std::vector<vec3> controlPoints = {
		vec3(1.0f, 0.5f, 0.5f),
		vec3(1.0f, 1.0f, 2.0f),
		vec3(-1.0f, 1.0f, 4.0f),
		vec3(-1.0f, -1.0f, 6.0f),
		vec3(0.0f, 0.0f, 8.0f)
	};

	// linear for now
	vec3 getPointByHeight(float z) const {
		auto it = controlPoints.begin();
		while(it->z < z) ++it;
		auto prev = it - 1;

		float r = ((z - prev->z) / (it->z - prev->z));

		float maxz = controlPoints.back().z;

		return ((1.0f - r) * (*prev) + r * (*it)) + vec3(cosf(t * 8.0f + z * 1.5f) * (maxz - z), sinf(t * 6.0f + z * 4.0f) * (maxz - z), 0);
	}

	vec3 getMinPoint() const {
		return getPointByHeight(controlPoints.front().z);
	}


	VertexData GenVertexData(float u, float v) {
		float min = controlPoints.front().z;
		float max = controlPoints.back().z;
		float z = u * (max - min) + min;
		vec3 center = getPointByHeight(z);
		VertexData vd;
		vd.position = center + vec3(cosf(v * 2 * M_PI), sinf(v * 2 * M_PI), z);
		vd.normal = (vd.position - center).normalize();
		vd.u = u;
		vd.v = v;
		return vd;
	}

	void Animate(float t) {
		this->t = t / 2.0f;
		Create(32, 10);
	}


};

Sphere *sphere;
Sphere *snakeHead;
Snake *snake;

class World {
	std::vector<Geometry *> objects;
	std::vector<LightSource> lightSources;
public:

	void Create() {
		sphere = new Sphere(vec3(0, 0, 1), 1.0f);
		snakeHead = new Sphere(vec3(0, 0, 1), 0.8f);
		snakeHead->material.ka = vec3(0.1f, 0.4f, 0.2f);
		snakeHead->material.kd = vec3(0.3f, 0.7f, 0.5f);
		snakeHead->material.ks = vec3(0.8f, 0.8f, 0.8f);
		snakeHead->material.shine = 30;
		snakeHead->tx = 0;
		snakeHead->tz = 0;

		objects.push_back(sphere);
		objects.push_back(snakeHead);
		objects.push_back(new Desert(100, 20));
		objects.push_back(new Footbridge(100, 5));
		objects.push_back(snake = new Snake());
		lightSources.emplace_back(vec4(3, 3, 3, 1), vec4(0.3, 0.3, 0.3, 1), vec4(1, 1, 1, 1));

	}

	void Draw() {
		for(auto *obj: objects) {
			for(auto ls: lightSources) {
				ls.SetUniforms(shaderProgram);
			}

			mat4 M = obj->GetModelMatrix();
			M.SetUniform(shaderProgram, "M");
			mat4 Minv = obj->GetModelInverseMatrix();
			Minv.SetUniform(shaderProgram, "Minv");
			camera.SetVPUniform();
			camera.SetWEyeUniform();
			obj->SetUniforms(shaderProgram);
			obj->Draw();
		}
	}

	void Animate(float t) {
		for(auto *obj: objects) {
			obj->Animate(t);
		}

		camera.wLookat = sphere->GetPosition();

		snakeHead->center = snake->getMinPoint();

		if(sphere->intersect(*snakeHead)) {
			sphere->setCollided();
		}

		lightSources.front().pos = snakeHead->center;
	}

	~World() noexcept {
		for(auto *obj: objects) {
			delete obj;
		}
	}
};

World world;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	// Create objects by setting up their vertex data on the GPU
	world.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.7f, 0.9f, 1.0f, 0.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	world.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int /*pX*/, int /*pY*/) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == ' ') sphere->moving = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int /*pX*/, int /*pY*/) {
	if (key == ' ') sphere->moving = false;
}

// Mouse click event
void onMouse(int /*button*/, int /*state*/, int /*pX*/, int /*pY*/) {
}

// Move mouse with key pressed
void onMouseMotion(int /*pX*/, int /*pY*/) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	if(lastTime > 0.0f) {
		camera.Animate(sec);
		world.Animate(sec);

		glutPostRedisplay();
	}
	lastTime = sec;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

