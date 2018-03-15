
#ifndef SHADER_CREATOR_H
#define SHADER_CRWATOR_H

#include <iostream>
#include <fstream>
#include <memory>

#include <GL\glew.h>
#include <GL\freeglut.h>

namespace Jex
{
	struct Shader
	{
		const char * fileName;
		GLenum       type;
		const char * source;
	};

	class ShaderCreator
	{

	public:

		ShaderCreator() = default;

		~ShaderCreator() = default;

		GLuint CreateShaderProgram(const char * vertexShader, const char * fragmentShader);

		GLuint UseShaderProgram(GLuint shaderPrgm = -1);

		GLuint GetShaderProgram() const
		{
			return m_currShaderPrgm;
		}

	protected:

		static std::string ReadShaderSource(const char * shaderFile)
		{
			std::ifstream shaderStream(shaderFile);
			std::string  shaderSource = "";

			if (shaderStream.is_open())
			{
				char c;
				while (shaderStream.get(c))
				{
					shaderSource += c;
				}

				shaderStream.close();
			}
			
			return shaderSource;
		}

	private:

		GLuint m_currShaderPrgm;

	};


}

#endif